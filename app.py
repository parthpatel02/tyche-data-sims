import streamlit as st
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Clean implementation of the core classes
class PrizePool:
    def __init__(self, annual_rate=0.04):
        self.balance = 0.0
        self.daily_rate = annual_rate / 365.0
        self.ledger = []

    def add_funds(self, amount, date, source):
        if amount > 0:
            self.balance += amount
            self.ledger.append({'date': date, 'amount': amount, 'source': source})
    
    def accrue_interest(self, date):
        interest_earned = self.balance * self.daily_rate
        if interest_earned > 0:
            self.balance += interest_earned
            self.ledger.append({'date': date, 'amount': interest_earned, 'source': 'pool_interest'})
        return interest_earned

class User:
    def __init__(self, user_id, initial_goal, initial_cycle_day, creation_date):
        self.user_id = user_id
        self.creation_date = creation_date
        self.goal_amount = initial_goal
        self.cycle_day = initial_cycle_day
        self.spillover_balance = 0.0
        self.cumulative_achieved_goal_amount = 0.0
        self.transactions = []
        self.cycle_history = []
        self.pending_goal_amount = None

    def add_transaction(self, date, amount):
        self.transactions.append({'date': date, 'amount': amount})
        self.transactions.sort(key=lambda x: x['date'])

    def get_balance(self, on_date):
        return sum(t['amount'] for t in self.transactions if t['date'] <= on_date)

@dataclass
class SimulationParams:
    num_users: int = 1000
    simulation_days: int = 365
    user_interest_rate: float = 0.04
    pool_interest_rate: float = 0.04
    min_goal: float = 50
    max_goal: float = 500
    initial_deposit_min: float = 500
    initial_deposit_max: float = 5000
    monthly_deposit_probability: float = 0.7
    monthly_deposit_min: float = 50
    monthly_deposit_max: float = 300
    withdrawal_probability: float = 0.15
    withdrawal_percentage_min: float = 0.1
    withdrawal_percentage_max: float = 0.5
    goal_change_probability: float = 0.05

    monthly_prize_multiplier: float = 0.12
    quarterly_prize_multiplier: float = 0.15
    
    # Growth and Churn Parameters
    monthly_growth_rate: float = 0.05  # 5% monthly growth
    monthly_churn_rate: float = 0.02   # 2% monthly churn

class UserBehaviorProfile:
    def __init__(self, profile_type: str, custom_params=None):
        self.profile_type = profile_type
        
        if custom_params:
            # Use custom parameters from UI
            if profile_type == "conservative":
                self.deposit_frequency = custom_params['conservative_deposit_freq']
                self.withdrawal_frequency = custom_params['conservative_withdrawal_freq']
                self.goal_change_frequency = custom_params['conservative_goal_change_freq']
                self.initial_deposit_range = (custom_params['conservative_initial_min'], custom_params['conservative_initial_max'])
                self.monthly_deposit_range = (custom_params['conservative_monthly_min'], custom_params['conservative_monthly_max'])
            elif profile_type == "average":
                self.deposit_frequency = custom_params['average_deposit_freq']
                self.withdrawal_frequency = custom_params['average_withdrawal_freq']
                self.goal_change_frequency = custom_params['average_goal_change_freq']
                self.initial_deposit_range = (custom_params['average_initial_min'], custom_params['average_initial_max'])
                self.monthly_deposit_range = (custom_params['average_monthly_min'], custom_params['average_monthly_max'])
            else:  # aggressive
                self.deposit_frequency = custom_params['aggressive_deposit_freq']
                self.withdrawal_frequency = custom_params['aggressive_withdrawal_freq']
                self.goal_change_frequency = custom_params['aggressive_goal_change_freq']
                self.initial_deposit_range = (custom_params['aggressive_initial_min'], custom_params['aggressive_initial_max'])
                self.monthly_deposit_range = (custom_params['aggressive_monthly_min'], custom_params['aggressive_monthly_max'])
        else:
            # Use default values
            if profile_type == "conservative":
                self.deposit_frequency = 0.3
                self.withdrawal_frequency = 0.2
                self.goal_change_frequency = 0.08
                self.initial_deposit_range = (200, 800)
                self.monthly_deposit_range = (30, 120)
            elif profile_type == "average":
                self.deposit_frequency = 0.5
                self.withdrawal_frequency = 0.25
                self.goal_change_frequency = 0.06
                self.initial_deposit_range = (500, 2000)
                self.monthly_deposit_range = (80, 350)
            else:  # aggressive
                self.deposit_frequency = 0.7
                self.withdrawal_frequency = 0.15
                self.goal_change_frequency = 0.04
                self.initial_deposit_range = (1000, 5000)
                self.monthly_deposit_range = (250, 700)

class MultiUserSimulator:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.prize_pool = PrizePool(annual_rate=params.pool_interest_rate)
        self.users = []
        self.user_daily_rate = params.user_interest_rate / 365.0
        self.start_date = datetime.date(2024, 1, 1)
        self.end_date = self.start_date + datetime.timedelta(days=params.simulation_days)
        
        # Growth and Churn tracking
        self.user_lifecycle_events = []  # Track all growth/churn events
        self.next_user_id = 0
        
    def generate_users(self, custom_behavior_params=None):
        """Generate users with unified behavior profiles"""
        if custom_behavior_params:
            # Use custom distribution from UI
            conservative_pct = custom_behavior_params.get('conservative_pct', 0.3)
            aggressive_pct = custom_behavior_params.get('aggressive_pct', 0.2)
            average_pct = max(0.1, 1.0 - conservative_pct - aggressive_pct)
        else:
            # Use default distribution
            conservative_pct = 0.3
            aggressive_pct = 0.2
            average_pct = 0.5
        
        # Determine user profile distribution
        num_conservative = int(self.params.num_users * conservative_pct)
        num_aggressive = int(self.params.num_users * aggressive_pct)
        num_average = self.params.num_users - num_conservative - num_aggressive
        
        profiles = (["conservative"] * num_conservative + 
                   ["average"] * num_average + 
                   ["aggressive"] * num_aggressive)
        random.shuffle(profiles)
        
        for i in range(self.params.num_users):
            # Stagger user creation over first 30 days
            creation_date = self.start_date + datetime.timedelta(days=random.randint(0, 30))
            initial_goal = random.uniform(self.params.min_goal, self.params.max_goal)
            cycle_day = random.randint(1, 28)  # Avoid month-end complications
            
            user = User(f"user_{i:04d}", initial_goal, cycle_day, creation_date)
            user.behavior_profile = UserBehaviorProfile(profiles[i], custom_behavior_params)
            
            # Transaction ranges are now part of the behavior profile
            user.initial_deposit_range = user.behavior_profile.initial_deposit_range
            user.monthly_deposit_range = user.behavior_profile.monthly_deposit_range
            
            self.users.append(user)

    def generate_user_actions(self, user: User) -> List[Dict]:
        """Generate all actions for a user over the simulation period"""
        actions = []
        
        # Initial deposit on creation date
        initial_amount = random.uniform(*user.initial_deposit_range)
        actions.append({
            'date': user.creation_date,
            'type': 'deposit',
            'value': initial_amount
        })
        
        # Generate monthly actions
        current_date = user.creation_date + relativedelta(months=1)
        while current_date <= self.end_date:
            # Monthly deposit
            if random.random() < user.behavior_profile.deposit_frequency:
                deposit_amount = random.uniform(*user.monthly_deposit_range)
                deposit_date = current_date + datetime.timedelta(days=random.randint(-10, 10))
                if deposit_date <= self.end_date:
                    actions.append({
                        'date': deposit_date,
                        'type': 'deposit',
                        'value': deposit_amount
                    })
            
            # Withdrawals
            if random.random() < user.behavior_profile.withdrawal_frequency:
                # Estimate current balance to determine withdrawal amount
                estimated_balance = sum(a['value'] if a['type'] == 'deposit' else -a['value'] 
                                      for a in actions if a['date'] <= current_date)
                if estimated_balance > 100:
                    withdrawal_pct = random.uniform(self.params.withdrawal_percentage_min, 
                                                  self.params.withdrawal_percentage_max)
                    withdrawal_amount = estimated_balance * withdrawal_pct
                    withdrawal_date = current_date + datetime.timedelta(days=random.randint(-10, 10))
                    if withdrawal_date <= self.end_date:
                        actions.append({
                            'date': withdrawal_date,
                            'type': 'withdrawal',
                            'value': withdrawal_amount
                        })
            
            # Goal changes
            if random.random() < user.behavior_profile.goal_change_frequency:
                new_goal = random.uniform(self.params.min_goal, self.params.max_goal)
                goal_change_date = current_date + datetime.timedelta(days=random.randint(-15, 15))
                if goal_change_date <= self.end_date:
                    actions.append({
                        'date': goal_change_date,
                        'type': 'change_goal',
                        'value': new_goal
                    })
            
            current_date += relativedelta(months=1)
        
        # Sort all actions by date
        actions.sort(key=lambda x: x['date'])
        return actions

    def get_cycle_period_for_date(self, user: User, for_date: datetime.date) -> Tuple[datetime.date, datetime.date]:
        """Calculate the cycle period that a given date falls into"""
        if for_date.day > user.cycle_day:
            temp_date = for_date + relativedelta(months=1)
        else:
            temp_date = for_date

        last_day_of_month = (datetime.date(temp_date.year, temp_date.month, 1) + 
                           relativedelta(months=1, days=-1)).day
        cycle_end_day = min(user.cycle_day, last_day_of_month)
        cycle_end_date = datetime.date(temp_date.year, temp_date.month, cycle_end_day)
        
        # Start date is one month prior
        temp_start_date = cycle_end_date - relativedelta(months=1)
        last_day_of_start_month = (datetime.date(temp_start_date.year, temp_start_date.month, 1) + 
                                 relativedelta(months=1, days=-1)).day
        cycle_start_day = min(user.cycle_day, last_day_of_start_month)
        cycle_start_date = datetime.date(temp_start_date.year, temp_start_date.month, cycle_start_day)
        
        return cycle_start_date, cycle_end_date

    def process_end_of_cycle(self, user: User, cycle_end_date: datetime.date):
        """Process end of cycle for a user (same logic as original)"""
        cycle_start_date, _ = self.get_cycle_period_for_date(user, cycle_end_date)
        
        initial_spillover_for_cycle = user.spillover_balance
        spillover_used = min(initial_spillover_for_cycle, user.goal_amount)
        user.spillover_balance = initial_spillover_for_cycle - spillover_used
        goal_remaining = user.goal_amount - spillover_used
        
        transactions_in_cycle = [t for t in user.transactions 
                               if cycle_start_date < t['date'] <= cycle_end_date]
        net_savings = sum(t['amount'] for t in transactions_in_cycle)
        
        goal_achieved = net_savings >= goal_remaining
        
        if goal_achieved:
            surplus = net_savings - goal_remaining
            user.spillover_balance += surplus
            user.cumulative_achieved_goal_amount += user.goal_amount
            
            if user.goal_amount > 0:
                effective_savings = spillover_used + net_savings
                achievement_pct = effective_savings / user.goal_amount
            else:
                achievement_pct = 1.0
        else:
            user.spillover_balance += net_savings
            if user.goal_amount > 0:
                achievement_pct = net_savings / user.goal_amount
            else:
                achievement_pct = 0.0

        user.spillover_balance = max(0, user.spillover_balance)
        user.cycle_history.append({'end_date': cycle_end_date, 'achievement_pct': achievement_pct})

        # Apply pending goal changes
        if user.pending_goal_amount is not None:
            user.goal_amount = user.pending_goal_amount
            user.pending_goal_amount = None

    def create_new_user(self, creation_date, custom_behavior_params=None):
        """Create a new user with random profile"""
        # Determine profile distribution (same logic as generate_users)
        if custom_behavior_params:
            conservative_pct = custom_behavior_params.get('conservative_pct', 0.3)
            aggressive_pct = custom_behavior_params.get('aggressive_pct', 0.2)
        else:
            conservative_pct = 0.3
            aggressive_pct = 0.2
            
        # Randomly assign profile based on distribution
        rand = random.random()
        if rand < conservative_pct:
            profile_type = "conservative"
        elif rand < conservative_pct + aggressive_pct:
            profile_type = "aggressive"
        else:
            profile_type = "average"
            
        # Create new user
        initial_goal = random.uniform(self.params.min_goal, self.params.max_goal)
        cycle_day = random.randint(1, 28)
        
        user = User(f"user_{self.next_user_id:04d}", initial_goal, cycle_day, creation_date)
        user.behavior_profile = UserBehaviorProfile(profile_type, custom_behavior_params)
        user.initial_deposit_range = user.behavior_profile.initial_deposit_range
        user.monthly_deposit_range = user.behavior_profile.monthly_deposit_range
        
        self.next_user_id += 1
        return user
        
    def churn_user(self, user, churn_date):
        """Process user churn with refund"""
        # Calculate user's current balance
        current_balance = user.get_balance(churn_date)
        
        # Issue refund (withdrawal)
        if current_balance > 0:
            user.add_transaction(churn_date, -current_balance)
            
        # Record churn event
        self.user_lifecycle_events.append({
            'date': churn_date,
            'event_type': 'churn',
            'user_id': user.user_id,
            'refund_amount': current_balance
        })
        
        return current_balance
        
    def process_daily_growth_churn(self, current_date, all_user_actions, user_action_indices, custom_behavior_params=None):
        """Process growth and churn for the current day"""
        # Calculate daily probabilities from monthly rates
        days_in_month = 30  # Approximate
        daily_growth_prob = self.params.monthly_growth_rate / days_in_month
        daily_churn_prob = self.params.monthly_churn_rate / days_in_month
        
        active_users = [u for u in self.users if u.user_id not in [event['user_id'] for event in self.user_lifecycle_events if event['event_type'] == 'churn']]
        
        # Process growth
        expected_new_users = len(active_users) * daily_growth_prob
        # Use Poisson-like approach for integer number of users
        num_new_users = np.random.poisson(expected_new_users) if expected_new_users > 0 else 0
        
        for _ in range(num_new_users):
            new_user = self.create_new_user(current_date, custom_behavior_params)
            self.users.append(new_user)
            
            # New user makes initial deposit
            initial_amount = random.uniform(*new_user.initial_deposit_range)
            new_user.add_transaction(current_date, initial_amount)
            
            # Generate actions for the new user
            all_user_actions[new_user.user_id] = self.generate_user_actions(new_user)
            user_action_indices[new_user.user_id] = 0
            
            # Record growth event
            self.user_lifecycle_events.append({
                'date': current_date,
                'event_type': 'growth',
                'user_id': new_user.user_id,
                'initial_deposit': initial_amount
            })
        
        # Process churn
        if len(active_users) > 0:
            expected_churned_users = len(active_users) * daily_churn_prob
            num_churned_users = min(np.random.poisson(expected_churned_users) if expected_churned_users > 0 else 0, len(active_users))
            
            if num_churned_users > 0:
                churned_users = random.sample(active_users, num_churned_users)
                for user in churned_users:
                    self.churn_user(user, current_date)

    def run_simulation(self, custom_behavior_params=None):
        """Run the complete simulation"""
        st.write("Generating users...")
        self.generate_users(custom_behavior_params)
        
        # Initialize next_user_id for new users
        self.next_user_id = len(self.users)
        
        st.write("Generating user actions...")
        all_user_actions = {}
        for user in self.users:
            all_user_actions[user.user_id] = self.generate_user_actions(user)
        
        st.write("Processing simulation day by day...")
        progress_bar = st.progress(0)
        
        # Create action queues for each user
        user_action_indices = {user.user_id: 0 for user in self.users}
        
        current_date = self.start_date
        total_days = (self.end_date - self.start_date).days
        day_count = 0
        
        while current_date <= self.end_date:
            # 1. Process growth and churn for this day
            self.process_daily_growth_churn(current_date, all_user_actions, user_action_indices, custom_behavior_params)
            
            # 2. Prize pool earns interest ONCE per day (not per user!)
            if current_date > self.start_date:
                self.prize_pool.accrue_interest(current_date)
            
            # 3. Get list of active users (not churned)
            churned_user_ids = {event['user_id'] for event in self.user_lifecycle_events if event['event_type'] == 'churn'}
            active_users = [user for user in self.users if user.user_id not in churned_user_ids]
            
            # 4. Process each active user for this day
            for user in active_users:
                if current_date >= user.creation_date:
                    # User contributes foregone interest to pool
                    if current_date > user.creation_date:
                        yesterday = current_date - datetime.timedelta(days=1)
                        balance_yesterday = user.get_balance(yesterday)
                        print(f"DEBUG: User {user.user_id}, yesterday={yesterday}, balance_yesterday={balance_yesterday}, daily_rate={self.user_daily_rate}, interest={balance_yesterday * self.user_daily_rate}")
                        print(f"  User {user.user_id} transactions: {[(t['date'], t['amount']) for t in user.transactions if t['date'] <= yesterday]}")
                        interest_foregone = balance_yesterday * self.user_daily_rate
                        self.prize_pool.add_funds(interest_foregone, current_date, f"user_{user.user_id}")
                    
                    # Process any user actions for this date
                    user_actions = all_user_actions[user.user_id]
                    action_idx = user_action_indices[user.user_id]
                    
                    while (action_idx < len(user_actions) and 
                           user_actions[action_idx]['date'] == current_date):
                        action = user_actions[action_idx]
                        
                        if action['type'] == 'deposit':
                            user.add_transaction(current_date, action['value'])
                        elif action['type'] == 'withdrawal':
                            user.add_transaction(current_date, -action['value'])
                        elif action['type'] == 'change_goal':
                            user.pending_goal_amount = action['value']
                        
                        action_idx += 1
                    
                    user_action_indices[user.user_id] = action_idx
                    
                    # Check for cycle end (yesterday's date)
                    if current_date > user.creation_date:
                        yesterday = current_date - datetime.timedelta(days=1)
                        last_day_of_yesterdays_month = (datetime.date(yesterday.year, yesterday.month, 1) + 
                                                      relativedelta(months=1, days=-1)).day
                        
                        is_cycle_end_day = (yesterday.day == user.cycle_day or 
                                          (user.cycle_day > last_day_of_yesterdays_month and 
                                           yesterday.day == last_day_of_yesterdays_month))
                        
                        if is_cycle_end_day:
                            self.process_end_of_cycle(user, yesterday)
            
            current_date += datetime.timedelta(days=1)
            day_count += 1
            
            # Update progress every 30 days
            if day_count % 30 == 0:
                progress_bar.progress(min(day_count / total_days, 1.0))
        
        progress_bar.empty()
        st.success(f"Simulation complete! Processed {len(self.users)} users over {day_count} days.")

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive business metrics"""
        total_balance = sum(user.get_balance(self.end_date) for user in self.users)
        total_achieved_goals = sum(user.cumulative_achieved_goal_amount for user in self.users)
        
        # User distribution
        conservative_savers = sum(1 for user in self.users if user.behavior_profile.profile_type == "conservative")
        average_savers = sum(1 for user in self.users if user.behavior_profile.profile_type == "average")
        aggressive_savers = len(self.users) - conservative_savers - average_savers
        
        # Prize pool analysis
        total_interest_contributed = sum(
            entry['amount'] for entry in self.prize_pool.ledger 
            if entry['source'].startswith('user_')
        )
        
        pool_interest_earned = sum(
            entry['amount'] for entry in self.prize_pool.ledger 
            if entry['source'] == 'pool_interest'
        )
        
        avg_cycles = np.mean([len(user.cycle_history) for user in self.users]) if self.users else 0
        avg_balance = total_balance / len(self.users) if self.users else 0
        avg_achieved_goals = total_achieved_goals / len(self.users) if self.users else 0
        
        return {
            'total_users': len(self.users),
            'total_balance': total_balance,
            'avg_balance_per_user': avg_balance,
            'total_achieved_goals': total_achieved_goals,
            'avg_achieved_goals_per_user': avg_achieved_goals,
            'prize_pool_balance': self.prize_pool.balance,
            'total_interest_contributed': total_interest_contributed,
            'pool_interest_earned': pool_interest_earned,
            'avg_cycles_completed': avg_cycles,
            'conservative_savers': conservative_savers,
            'average_savers': average_savers,
            'aggressive_savers': aggressive_savers,
        }

    def calculate_detailed_metrics(self) -> Dict:
        """Calculate comprehensive analytics for validation"""
        
        # Basic user stats
        total_users = len(self.users)
        if total_users == 0:
            return {}
        
        # 1. Goal Analysis
        current_goals = [user.goal_amount for user in self.users]
        avg_goal = np.mean(current_goals)
        goal_distribution = {
            'under_100': sum(1 for g in current_goals if g < 100),
            '100_200': sum(1 for g in current_goals if 100 <= g < 200),
            '200_300': sum(1 for g in current_goals if 200 <= g < 300),
            '300_400': sum(1 for g in current_goals if 300 <= g < 400),
            'over_400': sum(1 for g in current_goals if g >= 400)
        }
        
        # 2. Goal Achievement Analysis
        users_with_cycles = [user for user in self.users if user.cycle_history]
        total_cycles = sum(len(user.cycle_history) for user in users_with_cycles)
        
        if total_cycles > 0:
            # Achievement rates by user type
            achievement_stats = {
                'conservative': {'achieved': 0, 'total': 0},
                'average': {'achieved': 0, 'total': 0},
                'aggressive': {'achieved': 0, 'total': 0}
            }
            
            all_achievement_rates = []
            
            for user in users_with_cycles:
                user_achievements = [cycle['achievement_pct'] for cycle in user.cycle_history]
                avg_user_achievement = np.mean(user_achievements)
                all_achievement_rates.append(avg_user_achievement)
                
                profile = user.behavior_profile.profile_type
                achievement_stats[profile]['total'] += len(user_achievements)
                achievement_stats[profile]['achieved'] += sum(1 for rate in user_achievements if rate >= 1.0)
            
            overall_achievement_rate = np.mean(all_achievement_rates)
            
            # Calculate achievement rate by profile
            profile_achievement_rates = {}
            for profile, stats in achievement_stats.items():
                if stats['total'] > 0:
                    profile_achievement_rates[profile] = stats['achieved'] / stats['total']
                else:
                    profile_achievement_rates[profile] = 0.0
        else:
            overall_achievement_rate = 0.0
            profile_achievement_rates = {'conservative': 0.0, 'average': 0.0, 'aggressive': 0.0}
        
        # 3. Spillover Analysis
        spillover_balances = [user.spillover_balance for user in self.users]
        users_with_spillover = sum(1 for balance in spillover_balances if balance > 0)
        total_spillover = sum(spillover_balances)
        avg_spillover = np.mean(spillover_balances)
        max_spillover = max(spillover_balances) if spillover_balances else 0
        
        # 4. Transaction Analysis
        total_deposits = 0
        total_withdrawals = 0
        deposit_counts = []
        withdrawal_counts = []
        
        for user in self.users:
            user_deposits = sum(1 for t in user.transactions if t['amount'] > 0)
            user_withdrawals = sum(1 for t in user.transactions if t['amount'] < 0)
            
            deposit_counts.append(user_deposits)
            withdrawal_counts.append(user_withdrawals)
            
            total_deposits += sum(t['amount'] for t in user.transactions if t['amount'] > 0)
            total_withdrawals += sum(abs(t['amount']) for t in user.transactions if t['amount'] < 0)
        
        avg_deposits_per_user = np.mean(deposit_counts)
        avg_withdrawals_per_user = np.mean(withdrawal_counts)
        
        # 5. Balance Distribution
        final_balances = [user.get_balance(self.end_date) for user in self.users]
        balance_distribution = {
            'under_500': sum(1 for b in final_balances if b < 500),
            '500_1000': sum(1 for b in final_balances if 500 <= b < 1000),
            '1000_2000': sum(1 for b in final_balances if 1000 <= b < 2000),
            '2000_5000': sum(1 for b in final_balances if 2000 <= b < 5000),
            'over_5000': sum(1 for b in final_balances if b >= 5000)
        }
        
        # 6. Cumulative Achievement Analysis
        cumulative_goals = [user.cumulative_achieved_goal_amount for user in self.users]
        users_with_achievements = sum(1 for amount in cumulative_goals if amount > 0)
        total_cumulative = sum(cumulative_goals)
        avg_cumulative = np.mean(cumulative_goals)
        
        # 7. Prize Pool Health
        total_balance = sum(final_balances)
        pool_to_balance_ratio = self.prize_pool.balance / max(total_balance, 1)
        
        # Interest breakdown
        total_interest_contributed = sum(
            entry['amount'] for entry in self.prize_pool.ledger 
            if entry['source'].startswith('user_')
        )
        pool_interest_earned = sum(
            entry['amount'] for entry in self.prize_pool.ledger 
            if entry['source'] == 'pool_interest'
        )
        
        return {
            # Basic metrics
            'total_users': total_users,
            'simulation_days': self.params.simulation_days,
            
            # Goal metrics
            'avg_monthly_goal': avg_goal,
            'goal_distribution': goal_distribution,
            'current_goals': current_goals,  # For histogram
            
            # Achievement metrics
            'overall_achievement_rate': overall_achievement_rate,
            'profile_achievement_rates': profile_achievement_rates,
            'total_cycles_completed': total_cycles,
            'avg_cycles_per_user': total_cycles / total_users if total_users > 0 else 0,
            'all_achievement_rates': all_achievement_rates,  # For distribution
            
            # Spillover metrics
            'total_spillover': total_spillover,
            'avg_spillover': avg_spillover,
            'max_spillover': max_spillover,
            'users_with_spillover': users_with_spillover,
            'spillover_percentage': users_with_spillover / total_users * 100,
            'spillover_balances': spillover_balances,  # For histogram
            
            # Transaction metrics
            'total_deposits': total_deposits,
            'total_withdrawals': total_withdrawals,
            'net_flow': total_deposits - total_withdrawals,
            'avg_deposits_per_user': avg_deposits_per_user,
            'avg_withdrawals_per_user': avg_withdrawals_per_user,
            'deposit_counts': deposit_counts,  # For distribution
            'withdrawal_counts': withdrawal_counts,  # For distribution
            
            # Balance metrics
            'total_balance': total_balance,
            'avg_balance': total_balance / total_users,
            'balance_distribution': balance_distribution,
            'final_balances': final_balances,  # For histogram
            
            # Cumulative achievement metrics
            'total_cumulative_goals': total_cumulative,
            'avg_cumulative_goals': avg_cumulative,
            'users_with_achievements': users_with_achievements,
            'achievement_percentage': users_with_achievements / total_users * 100,
            'cumulative_goals': cumulative_goals,  # For histogram
            
            # Prize pool metrics
            'prize_pool_balance': self.prize_pool.balance,
            'pool_to_balance_ratio': pool_to_balance_ratio,
            'total_interest_contributed': total_interest_contributed,
            'pool_interest_earned': pool_interest_earned,
            
            # User distribution
            'conservative_savers': sum(1 for user in self.users if user.behavior_profile.profile_type == "conservative"),
            'average_savers': sum(1 for user in self.users if user.behavior_profile.profile_type == "average"),
            'aggressive_savers': sum(1 for user in self.users if user.behavior_profile.profile_type == "aggressive"),
        }

    def calculate_monthly_goal_achievements(self):
        """Calculate goal achievement statistics by month"""
        monthly_data = {}
        
        # Group cycle history by month
        for user in self.users:
            for cycle in user.cycle_history:
                cycle_end = cycle['end_date']
                month_key = f"{cycle_end.year}-{cycle_end.month:02d}"
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        'month': cycle_end.strftime('%B %Y'),
                        'total_cycles': 0,
                        'goals_achieved': 0,
                        'goals_failed': 0,
                        'total_goal_amount': 0,
                        'total_achieved_amount': 0,
                        'users_with_cycles': set(),
                        'achievement_rates': [],
                        'goal_amounts': []
                    }
                
                monthly_data[month_key]['total_cycles'] += 1
                monthly_data[month_key]['users_with_cycles'].add(user.user_id)
                monthly_data[month_key]['achievement_rates'].append(cycle['achievement_pct'])
                monthly_data[month_key]['goal_amounts'].append(user.goal_amount)
                
                if cycle['achievement_pct'] >= 1.0:
                    monthly_data[month_key]['goals_achieved'] += 1
                    monthly_data[month_key]['total_achieved_amount'] += user.goal_amount
                else:
                    monthly_data[month_key]['goals_failed'] += 1
                
                monthly_data[month_key]['total_goal_amount'] += user.goal_amount
        
        # Convert to list and calculate final metrics
        monthly_results = []
        for month_key in sorted(monthly_data.keys()):
            data = monthly_data[month_key]
            
            achievement_rate = data['goals_achieved'] / data['total_cycles'] if data['total_cycles'] > 0 else 0
            avg_achievement_pct = np.mean(data['achievement_rates']) if data['achievement_rates'] else 0
            avg_goal_amount = np.mean(data['goal_amounts']) if data['goal_amounts'] else 0
            
            monthly_results.append({
                'month': data['month'],
                'month_key': month_key,
                'unique_users': len(data['users_with_cycles']),
                'total_cycles': data['total_cycles'],
                'goals_achieved': data['goals_achieved'],
                'goals_failed': data['goals_failed'],
                'achievement_rate': achievement_rate,
                'avg_achievement_pct': avg_achievement_pct,
                'avg_goal_amount': avg_goal_amount,
                'total_goal_amount': data['total_goal_amount'],
                'total_achieved_amount': data['total_achieved_amount'],
                'achievement_rates': data['achievement_rates']
            })
        
        return monthly_results

    def calculate_monthly_financial_metrics(self):
        """Calculate monthly financial summary metrics"""
        monthly_data = {}
        
        # Calculate metrics for each month in the simulation period
        current_date = self.start_date
        while current_date <= self.end_date:
            month_key = f"{current_date.year}-{current_date.month:02d}"
            
            if month_key not in monthly_data:
                # Find start and end of this month
                month_start = datetime.date(current_date.year, current_date.month, 1)
                if current_date.month == 12:
                    month_end = datetime.date(current_date.year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    month_end = datetime.date(current_date.year, current_date.month + 1, 1) - datetime.timedelta(days=1)
                
                # Ensure we don't go beyond simulation end
                month_end = min(month_end, self.end_date)
                
                monthly_data[month_key] = {
                    'month': current_date.strftime('%B %Y'),
                    'month_start': month_start,
                    'month_end': month_end,
                    'daily_balances': [],
                    'total_final_balance': 0,
                    'avg_daily_balance': 0,
                    'interest_earned': 0,
                    'prize_pool_total': 0
                }
                
                # Calculate total user balance at end of month
                total_end_balance = sum(user.get_balance(month_end) for user in self.users)
                monthly_data[month_key]['total_final_balance'] = total_end_balance
                
                # Calculate average daily balance for the month
                daily_balances = []
                check_date = month_start
                while check_date <= month_end:
                    daily_total = sum(user.get_balance(check_date) for user in self.users if check_date >= user.creation_date)
                    daily_balances.append(daily_total)
                    check_date += datetime.timedelta(days=1)
                
                if daily_balances:
                    monthly_data[month_key]['avg_daily_balance'] = np.mean(daily_balances)
                
                # Calculate interest earned during this month (contributed to prize pool)
                month_interest = sum(
                    entry['amount'] for entry in self.prize_pool.ledger
                    if (entry['source'].startswith('user_') and 
                        month_start <= entry['date'] <= month_end)
                )
                monthly_data[month_key]['interest_earned'] = month_interest
                
                # Get prize pool balance at end of month
                pool_balance_end = sum(
                    entry['amount'] for entry in self.prize_pool.ledger 
                    if entry['date'] <= month_end
                )
                monthly_data[month_key]['prize_pool_total'] = pool_balance_end
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime.date(current_date.year + 1, 1, 1)
            else:
                current_date = datetime.date(current_date.year, current_date.month + 1, 1)
        
        # Convert to sorted list
        monthly_results = []
        for month_key in sorted(monthly_data.keys()):
            data = monthly_data[month_key]
            monthly_results.append({
                'month': data['month'],
                'month_key': month_key,
                'total_final_balance': data['total_final_balance'],
                'avg_daily_balance': data['avg_daily_balance'],
                'interest_earned': data['interest_earned'],
                'prize_pool_total': data['prize_pool_total']
            })
        
        return monthly_results

    def calculate_monthly_lifecycle_metrics(self):
        """Calculate monthly user lifecycle and financial impact metrics"""
        monthly_data = {}
        
        # Process lifecycle events by month
        for event in self.user_lifecycle_events:
            month_key = f"{event['date'].year}-{event['date'].month:02d}"
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'month': month_key,
                    'new_users': 0,
                    'churned_users': 0,
                    'churn_refunds': 0.0,
                    'new_user_deposits': 0.0
                }
            
            if event['event_type'] == 'growth':
                monthly_data[month_key]['new_users'] += 1
                monthly_data[month_key]['new_user_deposits'] += event.get('initial_deposit', 0)
            elif event['event_type'] == 'churn':
                monthly_data[month_key]['churned_users'] += 1
                monthly_data[month_key]['churn_refunds'] += event.get('refund_amount', 0)
        
        # Calculate net metrics
        for month_key in monthly_data:
            data = monthly_data[month_key]
            data['net_user_growth'] = data['new_users'] - data['churned_users']
            data['net_balance_impact'] = data['new_user_deposits'] - data['churn_refunds']
        
        return monthly_data

    def display_monthly_lifecycle_summary(self):
        """Display monthly user lifecycle and financial impact summary"""
        if not self.user_lifecycle_events:
            st.info("No user lifecycle events occurred during simulation.")
            return
            
        monthly_data = self.calculate_monthly_lifecycle_metrics()
        
        if not monthly_data:
            st.info("No monthly lifecycle data available.")
            return
        
        st.subheader("👥 Monthly User Lifecycle & Financial Impact")
        
        # Convert to DataFrame
        df = pd.DataFrame(list(monthly_data.values()))
        df = df.sort_values('month')
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_new_users = df['new_users'].sum()
        total_churned_users = df['churned_users'].sum()
        total_refunds = df['churn_refunds'].sum()
        total_new_deposits = df['new_user_deposits'].sum()
        
        with col1:
            st.metric("Total New Users", f"{total_new_users:,}")
            st.metric("Total Churned Users", f"{total_churned_users:,}")
        
        with col2:
            net_user_growth = total_new_users - total_churned_users
            st.metric("Net User Growth", f"{net_user_growth:,}")
            
            if len(self.users) > 0:
                growth_rate = (net_user_growth / len(self.users)) * 100
                st.metric("Net Growth Rate", f"{growth_rate:.1f}%")
        
        with col3:
            st.metric("Total Churn Refunds", f"€{total_refunds:,.2f}")
            st.metric("Total New Deposits", f"€{total_new_deposits:,.2f}")
        
        with col4:
            net_balance_impact = total_new_deposits - total_refunds
            st.metric("Net Balance Impact", f"€{net_balance_impact:,.2f}")
            
            avg_refund = total_refunds / total_churned_users if total_churned_users > 0 else 0
            st.metric("Avg Refund per Churn", f"€{avg_refund:,.2f}")
        
        # Display monthly table
        st.subheader("📊 Monthly Breakdown")
        
        # Format the DataFrame for display
        display_df = df.copy()
        display_df['New Users'] = display_df['new_users'].astype(int)
        display_df['Churned Users'] = display_df['churned_users'].astype(int)
        display_df['Net Growth'] = display_df['net_user_growth'].astype(int)
        display_df['Churn Refunds'] = display_df['churn_refunds'].apply(lambda x: f"€{x:,.2f}")
        display_df['New Deposits'] = display_df['new_user_deposits'].apply(lambda x: f"€{x:,.2f}")
        display_df['Net Impact'] = display_df['net_balance_impact'].apply(lambda x: f"€{x:,.2f}")
        display_df['Month'] = display_df['month']
        
        # Select and reorder columns for display
        display_columns = ['Month', 'New Users', 'Churned Users', 'Net Growth', 
                          'New Deposits', 'Churn Refunds', 'Net Impact']
        display_df = display_df[display_columns]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # User Growth/Churn Chart
            fig_users = go.Figure()
            fig_users.add_trace(go.Bar(
                name='New Users',
                x=df['month'],
                y=df['new_users'],
                marker_color='green',
                opacity=0.7
            ))
            fig_users.add_trace(go.Bar(
                name='Churned Users',
                x=df['month'],
                y=df['churned_users'],
                marker_color='red',
                opacity=0.7
            ))
            fig_users.update_layout(
                title="Monthly User Growth vs Churn",
                xaxis_title="Month",
                yaxis_title="Number of Users",
                barmode='group'
            )
            st.plotly_chart(fig_users, use_container_width=True)
        
        with col2:
            # Financial Impact Chart
            fig_money = go.Figure()
            fig_money.add_trace(go.Bar(
                name='New Deposits',
                x=df['month'],
                y=df['new_user_deposits'],
                marker_color='blue',
                opacity=0.7
            ))
            fig_money.add_trace(go.Bar(
                name='Churn Refunds',
                x=df['month'],
                y=df['churn_refunds'],
                marker_color='orange',
                opacity=0.7
            ))
            fig_money.update_layout(
                title="Monthly Financial Impact",
                xaxis_title="Month",
                yaxis_title="Amount (€)",
                barmode='group'
            )
            st.plotly_chart(fig_money, use_container_width=True)

    def display_monthly_financial_summary(self):
        """Display monthly financial summary table and analytics"""
        st.subheader("💰 Monthly Financial Summary")
        st.write("Track key financial metrics month by month")
        
        monthly_data = self.calculate_monthly_financial_metrics()
        
        if not monthly_data:
            st.warning("No financial data available yet.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        final_month = monthly_data[-1] if monthly_data else {}
        total_interest = sum(month['interest_earned'] for month in monthly_data)
        
        with col1:
            st.metric("Current Total Balance", f"€{final_month.get('total_final_balance', 0):,.2f}")
        with col2:
            st.metric("Current Prize Pool", f"€{final_month.get('prize_pool_total', 0):,.2f}")
        with col3:
            st.metric("Total Interest Generated", f"€{total_interest:,.2f}")
        with col4:
            if len(monthly_data) >= 2:
                growth_rate = (final_month.get('total_final_balance', 0) - monthly_data[0]['total_final_balance']) / max(monthly_data[0]['total_final_balance'], 1)
                st.metric("Balance Growth Rate", f"{growth_rate:.1%}")
            else:
                st.metric("Balance Growth Rate", "N/A")
        
        # Monthly financial table
        st.subheader("📊 Monthly Breakdown")
        
        table_data = []
        for month in monthly_data:
            table_data.append({
                'Month': month['month'],
                'Total Final Balance': f"€{month['total_final_balance']:,.2f}",
                'Avg Daily Balance': f"€{month['avg_daily_balance']:,.2f}",
                'Interest Earned': f"€{month['interest_earned']:,.2f}",
                'Prize Pool Total': f"€{month['prize_pool_total']:,.2f}"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Financial trend charts
        st.subheader("📈 Financial Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Balance trends
            fig_balances = go.Figure()
            
            fig_balances.add_trace(go.Scatter(
                x=[month['month'] for month in monthly_data],
                y=[month['total_final_balance'] for month in monthly_data],
                mode='lines+markers',
                name='Total Final Balance',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig_balances.add_trace(go.Scatter(
                x=[month['month'] for month in monthly_data],
                y=[month['avg_daily_balance'] for month in monthly_data],
                mode='lines+markers',
                name='Avg Daily Balance',
                line=dict(color='lightblue', width=2),
                marker=dict(size=6)
            ))
            
            fig_balances.update_layout(
                title="User Balance Trends",
                xaxis_title="Month",
                yaxis_title="Balance (€)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_balances, use_container_width=True, key="monthly_balance_trends")
        
        with col2:
            # Interest and prize pool trends
            fig_pool = go.Figure()
            
            fig_pool.add_trace(go.Scatter(
                x=[month['month'] for month in monthly_data],
                y=[month['interest_earned'] for month in monthly_data],
                mode='lines+markers',
                name='Monthly Interest Earned',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            
            fig_pool.add_trace(go.Scatter(
                x=[month['month'] for month in monthly_data],
                y=[month['prize_pool_total'] for month in monthly_data],
                mode='lines+markers',
                name='Prize Pool Total',
                line=dict(color='gold', width=2),
                marker=dict(size=6),
                yaxis='y2'
            ))
            
            fig_pool.update_layout(
                title="Interest & Prize Pool Trends",
                xaxis_title="Month",
                yaxis=dict(title="Interest Earned (€)", side="left"),
                yaxis2=dict(title="Prize Pool Total (€)", side="right", overlaying="y"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_pool, use_container_width=True, key="monthly_interest_pool_trends")
        
        # Financial insights
        st.subheader("💡 Financial Insights")
        
        if len(monthly_data) >= 2:
            # Balance growth analysis
            latest_balance = monthly_data[-1]['total_final_balance']
            previous_balance = monthly_data[-2]['total_final_balance']
            balance_change = latest_balance - previous_balance
            balance_growth_rate = balance_change / max(previous_balance, 1)
            
            if balance_growth_rate > 0.05:
                st.success(f"📈 Strong balance growth: +{balance_growth_rate:.1%} (+€{balance_change:,.2f}) from last month")
            elif balance_growth_rate > 0:
                st.info(f"📊 Positive balance growth: +{balance_growth_rate:.1%} (+€{balance_change:,.2f}) from last month")
            else:
                st.warning(f"📉 Balance decline: {balance_growth_rate:.1%} (€{balance_change:,.2f}) from last month")
            
            # Interest generation analysis
            latest_interest = monthly_data[-1]['interest_earned']
            previous_interest = monthly_data[-2]['interest_earned']
            interest_change = latest_interest - previous_interest
            
            if interest_change > 0:
                st.info(f"💰 Interest generation increased by €{interest_change:,.2f} compared to last month")
            elif interest_change < 0:
                st.warning(f"💸 Interest generation decreased by €{abs(interest_change):,.2f} compared to last month")
            else:
                st.info("💰 Interest generation remained stable compared to last month")
            
            # Prize pool sustainability
            pool_growth_rate = (monthly_data[-1]['prize_pool_total'] - monthly_data[0]['prize_pool_total']) / max(monthly_data[0]['prize_pool_total'], 1)
            balance_growth_rate_total = (monthly_data[-1]['total_final_balance'] - monthly_data[0]['total_final_balance']) / max(monthly_data[0]['total_final_balance'], 1)
            
            if pool_growth_rate > balance_growth_rate_total:
                st.success("✅ **Healthy Growth**: Prize pool is growing faster than user balances - excellent sustainability!")
            elif pool_growth_rate > balance_growth_rate_total * 0.8:
                st.info("📊 **Balanced Growth**: Prize pool growth is keeping pace with user balance growth")
            else:
                st.warning("⚠️ **Monitor Growth**: Prize pool growth is lagging behind user balance growth")

    def display_monthly_goal_analysis(self):
        """Display monthly goal achievement analysis"""
        st.subheader("📊 Monthly Goal Achievement Analysis")
        st.write("Track how users perform against their savings goals each month")
        
        monthly_data = self.calculate_monthly_goal_achievements()
        
        if not monthly_data:
            st.warning("No goal achievement data available yet.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_cycles = sum(month['total_cycles'] for month in monthly_data)
        total_achieved = sum(month['goals_achieved'] for month in monthly_data)
        overall_rate = total_achieved / total_cycles if total_cycles > 0 else 0
        
        with col1:
            st.metric("Total Cycles Completed", f"{total_cycles:,}")
        with col2:
            st.metric("Goals Achieved", f"{total_achieved:,}")
        with col3:
            st.metric("Overall Achievement Rate", f"{overall_rate:.1%}")
        with col4:
            latest_month = monthly_data[-1] if monthly_data else {'achievement_rate': 0}
            st.metric("Latest Month Rate", f"{latest_month['achievement_rate']:.1%}")
        
        # Monthly trend table
        st.subheader("📅 Monthly Breakdown")
        
        table_data = []
        for month in monthly_data:
            table_data.append({
                'Month': month['month'],
                'Users': f"{month['unique_users']:,}",
                'Cycles': f"{month['total_cycles']:,}",
                'Achieved': f"{month['goals_achieved']:,}",
                'Failed': f"{month['goals_failed']:,}",
                'Achievement Rate': f"{month['achievement_rate']:.1%}",
                'Avg Achievement %': f"{month['avg_achievement_pct']:.1%}",
                'Avg Goal': f"€{month['avg_goal_amount']:.0f}",
                'Total Goal Amount': f"€{month['total_goal_amount']:,.0f}",
                'Achieved Amount': f"€{month['total_achieved_amount']:,.0f}"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Achievement rate trend
            fig_trend = px.line(
                x=[month['month'] for month in monthly_data],
                y=[month['achievement_rate'] * 100 for month in monthly_data],
                title="Monthly Achievement Rate Trend",
                labels={'x': 'Month', 'y': 'Achievement Rate (%)'},
                markers=True
            )
            fig_trend.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_trend, use_container_width=True, key="monthly_achievement_trend")
        
        with col2:
            # Users vs Cycles
            fig_users = px.bar(
                x=[month['month'] for month in monthly_data],
                y=[month['unique_users'] for month in monthly_data],
                title="Active Users per Month",
                labels={'x': 'Month', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig_users, use_container_width=True, key="monthly_users")
        
        # Achievement distribution
        st.subheader("🎯 Achievement Distribution by Month")
        
        # Create stacked bar chart data
        months = [month['month'] for month in monthly_data]
        achieved = [month['goals_achieved'] for month in monthly_data]
        failed = [month['goals_failed'] for month in monthly_data]
        
        fig_stacked = go.Figure(data=[
            go.Bar(name='Goals Achieved', x=months, y=achieved, marker_color='green'),
            go.Bar(name='Goals Failed', x=months, y=failed, marker_color='red')
        ])
        
        fig_stacked.update_layout(
            barmode='stack',
            title="Goals Achieved vs Failed by Month",
            xaxis_title="Month",
            yaxis_title="Number of Goals",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_stacked, use_container_width=True, key="monthly_stacked_goals")
        
        # Detailed breakdown for selected month
        st.subheader("🔍 Monthly Detail View")
        
        selected_month = st.selectbox(
            "Select month for detailed view:",
            options=[month['month'] for month in monthly_data],
            index=len(monthly_data) - 1 if monthly_data else 0
        )
        
        if selected_month:
            month_data = next((m for m in monthly_data if m['month'] == selected_month), None)
            
            if month_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Users in Month", f"{month_data['unique_users']:,}")
                    st.metric("Cycles Completed", f"{month_data['total_cycles']:,}")
                
                with col2:
                    st.metric("Goals Achieved", f"{month_data['goals_achieved']:,}")
                    st.metric("Goals Failed", f"{month_data['goals_failed']:,}")
                
                with col3:
                    st.metric("Achievement Rate", f"{month_data['achievement_rate']:.1%}")
                    st.metric("Avg Goal Amount", f"€{month_data['avg_goal_amount']:.0f}")
                
                # Achievement distribution histogram
                if month_data['achievement_rates']:
                    achievement_pcts = [rate * 100 for rate in month_data['achievement_rates']]
                    
                    fig_hist = px.histogram(
                        x=achievement_pcts,
                        nbins=20,
                        title=f"Achievement Distribution - {selected_month}",
                        labels={'x': 'Achievement Percentage', 'y': 'Number of Users'},
                        color_discrete_sequence=['skyblue']
                    )
                    
                    # Add vertical line at 100% (goal achievement)
                    fig_hist.add_vline(x=100, line_dash="dash", line_color="red", 
                                      annotation_text="Goal Target (100%)")
                    
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"achievement_dist_{month_data['month_key']}")
                    
                    # Statistics
                    st.write("**Achievement Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**Min:** {min(achievement_pcts):.1f}%")
                    with col2:
                        st.write(f"**Max:** {max(achievement_pcts):.1f}%")
                    with col3:
                        st.write(f"**Median:** {np.median(achievement_pcts):.1f}%")
                    with col4:
                        above_target = sum(1 for rate in achievement_pcts if rate >= 100)
                        st.write(f"**≥100%:** {above_target}/{len(achievement_pcts)} ({above_target/len(achievement_pcts):.1%})")

        # User behavior insights
        st.subheader("💡 Insights")
        
        if len(monthly_data) >= 2:
            # Trend analysis
            recent_rate = monthly_data[-1]['achievement_rate']
            previous_rate = monthly_data[-2]['achievement_rate']
            trend = recent_rate - previous_rate
            
            if trend > 0.05:
                st.success(f"📈 Improving trend: Achievement rate increased by {trend:.1%} from last month")
            elif trend < -0.05:
                st.warning(f"📉 Declining trend: Achievement rate decreased by {abs(trend):.1%} from last month")
            else:
                st.info(f"📊 Stable trend: Achievement rate changed by {trend:+.1%} from last month")
            
            # Goal difficulty analysis
            avg_goal_trend = monthly_data[-1]['avg_goal_amount'] - monthly_data[-2]['avg_goal_amount']
            if abs(avg_goal_trend) > 10:
                direction = "increased" if avg_goal_trend > 0 else "decreased"
                st.info(f"🎯 Goal amounts have {direction} by €{abs(avg_goal_trend):.0f} on average")
        
            # Recommendations
            overall_avg_achievement = np.mean([month['avg_achievement_pct'] for month in monthly_data])
            
            if overall_avg_achievement < 0.7:
                st.warning("⚠️ **Recommendation:** Average achievement rate is below 70%. Consider lowering default goal amounts or improving user engagement.")
            elif overall_avg_achievement > 0.9:
                st.info("💡 **Recommendation:** Users are achieving goals easily. Consider encouraging higher goal amounts for better savings outcomes.")
            else:
                st.success("✅ **Status:** Goal achievement rates are in a healthy range (70-90%).")

    def display_detailed_analytics(self, metrics, unique_prefix=""):
        """Display comprehensive analytics dashboard"""
        
        st.header("📊 Detailed Simulation Analytics")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", f"{metrics['total_users']:,}")
            st.metric("Avg Monthly Goal", f"€{metrics['avg_monthly_goal']:.2f}")
        with col2:
            st.metric("Overall Achievement Rate", f"{metrics['overall_achievement_rate']:.1%}")
            st.metric("Total Cycles Completed", f"{metrics['total_cycles_completed']:,}")
        with col3:
            st.metric("Users with Spillover", f"{metrics['users_with_spillover']:,} ({metrics['spillover_percentage']:.1f}%)")
            st.metric("Avg Spillover", f"€{metrics['avg_spillover']:.2f}")
        with col4:
            st.metric("Net Money Flow", f"€{metrics['net_flow']:,.2f}")
            st.metric("Pool/Balance Ratio", f"{metrics['pool_to_balance_ratio']:.2%}")
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            # Goal distribution histogram
            st.subheader("💰 Goal Distribution")
            fig_goals = px.histogram(
                x=metrics['current_goals'], 
                nbins=20,
                title="Distribution of Current Monthly Goals",
                labels={'x': 'Goal Amount (€)', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig_goals, use_container_width=True, key=f"{unique_prefix}_goals_histogram")
            
            # Achievement rate by profile
            st.subheader("🎯 Achievement Rates by Profile")
            profile_data = pd.DataFrame([
                {'Profile': 'Conservative Savers', 'Rate': metrics['profile_achievement_rates']['conservative']},
                {'Profile': 'Average Savers', 'Rate': metrics['profile_achievement_rates']['average']},
                {'Profile': 'Aggressive Savers', 'Rate': metrics['profile_achievement_rates']['aggressive']}
            ])
            fig_achievement = px.bar(
                profile_data, 
                x='Profile', 
                y='Rate',
                title="Goal Achievement Rate by User Type",
                labels={'Rate': 'Achievement Rate'},
                color='Profile'
            )
            fig_achievement.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_achievement, use_container_width=True, key=f"{unique_prefix}_achievement_by_profile")
        
        with col2:
            # Balance distribution
            st.subheader("💳 Final Balance Distribution")
            fig_balance = px.histogram(
                x=metrics['final_balances'], 
                nbins=25,
                title="Distribution of Final User Balances",
                labels={'x': 'Balance (€)', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig_balance, use_container_width=True, key=f"{unique_prefix}_balance_histogram")
            
            # Spillover distribution
            st.subheader("🔄 Spillover Balance Distribution")
            spillover_data = [s for s in metrics['spillover_balances'] if s > 0]  # Only show users with spillover
            if spillover_data:
                fig_spillover = px.histogram(
                    x=spillover_data, 
                    nbins=15,
                    title="Distribution of Spillover Balances (Users with Spillover > 0)",
                    labels={'x': 'Spillover Amount (€)', 'y': 'Number of Users'}
                )
                st.plotly_chart(fig_spillover, use_container_width=True, key=f"{unique_prefix}_spillover_histogram")
            else:
                st.info("No users have spillover balances > 0")
        
        # Transaction analysis
        st.subheader("💸 Transaction Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_deposits = px.histogram(
                x=metrics['deposit_counts'], 
                nbins=15,
                title="Deposits per User",
                labels={'x': 'Number of Deposits', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig_deposits, use_container_width=True, key=f"{unique_prefix}_deposits_histogram")
        
        with col2:
            fig_withdrawals = px.histogram(
                x=metrics['withdrawal_counts'], 
                nbins=15,
                title="Withdrawals per User",
                labels={'x': 'Number of Withdrawals', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig_withdrawals, use_container_width=True, key=f"{unique_prefix}_withdrawals_histogram")
        
        with col3:
            # Achievement distribution
            if metrics['all_achievement_rates']:
                fig_ach_dist = px.histogram(
                    x=metrics['all_achievement_rates'], 
                    nbins=20,
                    title="Achievement Rate Distribution",
                    labels={'x': 'Average Achievement Rate', 'y': 'Number of Users'}
                )
                fig_ach_dist.update_layout(xaxis_tickformat='.1%')
                st.plotly_chart(fig_ach_dist, use_container_width=True, key=f"{unique_prefix}_achievement_distribution")
        
        # Validation checks
        st.subheader("✅ Validation Checks")
        
        checks = []
        
        # Check if achievement rates match expected profiles
        conservative_rate = metrics['profile_achievement_rates']['conservative']
        average_rate = metrics['profile_achievement_rates']['average']
        aggressive_rate = metrics['profile_achievement_rates']['aggressive']
        
        if aggressive_rate > average_rate > conservative_rate:
            checks.append("✅ Achievement rates follow expected pattern (Aggressive > Average > Conservative)")
        else:
            checks.append("❌ Achievement rates don't follow expected pattern")
        
        # Check spillover usage
        if metrics['users_with_spillover'] > 0:
            checks.append(f"✅ {metrics['spillover_percentage']:.1f}% of users have spillover (good)")
        else:
            checks.append("⚠️ No users have spillover - check goal difficulty")
        
        # Check transaction patterns
        if metrics['avg_deposits_per_user'] > metrics['avg_withdrawals_per_user']:
            checks.append("✅ Users deposit more frequently than they withdraw")
        else:
            checks.append("❌ Users withdraw as much or more than they deposit")
        
        # Check goal distribution
        if 50 <= metrics['avg_monthly_goal'] <= 500:
            checks.append("✅ Average goals are in reasonable range")
        else:
            checks.append("⚠️ Average goals might be too high or low")
        
        # Check prize pool sustainability
        if metrics['pool_to_balance_ratio'] > 0.02:  # 2%
            checks.append("✅ Prize pool is healthy (>2% of total balance)")
        elif metrics['pool_to_balance_ratio'] > 0.01:  # 1%
            checks.append("⚠️ Prize pool is moderate (1-2% of total balance)")
        else:
            checks.append("❌ Prize pool might be too small (<1% of total balance)")
        
        for check in checks:
            st.write(check)
        
        # Summary table
        st.subheader("📋 Summary Statistics")
        summary_data = {
            'Metric': [
                'Average Monthly Goal', 'Overall Achievement Rate', 'Conservative Rate', 
                'Average Rate', 'Aggressive Rate', 'Users with Spillover',
                'Average Spillover', 'Average Deposits/User', 'Average Withdrawals/User',
                'Total Prize Pool', 'Prize Pool Ratio'
            ],
            'Value': [
                f"€{metrics['avg_monthly_goal']:.2f}",
                f"{metrics['overall_achievement_rate']:.1%}",
                f"{metrics['profile_achievement_rates']['conservative']:.1%}",
                f"{metrics['profile_achievement_rates']['average']:.1%}",
                f"{metrics['profile_achievement_rates']['aggressive']:.1%}",
                f"{metrics['users_with_spillover']:,} ({metrics['spillover_percentage']:.1f}%)",
                f"€{metrics['avg_spillover']:.2f}",
                f"{metrics['avg_deposits_per_user']:.1f}",
                f"{metrics['avg_withdrawals_per_user']:.1f}",
                f"€{metrics['prize_pool_balance']:,.2f}",
                f"{metrics['pool_to_balance_ratio']:.2%}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    def generate_daily_extract(self, num_days_to_show=10):
        """Generate detailed daily extract using actual simulation data"""
        daily_data = []
        
        current_date = self.start_date
        
        for day_count in range(min(num_days_to_show, self.params.simulation_days)):
            current_date = self.start_date + datetime.timedelta(days=day_count)
            yesterday = current_date - datetime.timedelta(days=1)
            
            daily_record = {
                'date': current_date,
                'day_number': day_count + 1,
                'pool_balance_start': 0,
                'pool_interest_earned': 0,
                'total_user_contributions': 0,
                'total_deposits': 0,
                'total_withdrawals': 0,
                'users_active': 0,
                'user_details': [],
                'pool_balance_end': 0,
                'debug_info': {}
            }
            
            # Calculate pool balances from ledger
            pool_start = sum(
                entry['amount'] for entry in self.prize_pool.ledger 
                if entry['date'] < current_date
            )
            
            pool_interest_today = sum(
                entry['amount'] for entry in self.prize_pool.ledger 
                if entry['date'] == current_date and entry['source'] == 'pool_interest'
            )
            
            user_contributions_today = sum(
                entry['amount'] for entry in self.prize_pool.ledger 
                if entry['date'] == current_date and entry['source'].startswith('user_')
            )
            
            pool_end = sum(
                entry['amount'] for entry in self.prize_pool.ledger 
                if entry['date'] <= current_date
            )
            
            daily_record['pool_balance_start'] = pool_start
            daily_record['pool_interest_earned'] = pool_interest_today
            daily_record['total_user_contributions'] = user_contributions_today
            daily_record['pool_balance_end'] = pool_end
            
            # Calculate TOTAL deposits/withdrawals across ALL users for this day
            total_daily_deposits = 0
            total_daily_withdrawals = 0
            active_users_count = 0
            users_with_interest = 0
            total_balance_start = 0
            total_balance_end = 0
            
            for user in self.users:
                if current_date >= user.creation_date:
                    # Get transactions for this specific day
                    day_transactions = [t for t in user.transactions if t['date'] == current_date]
                    
                    day_deposits = sum(t['amount'] for t in day_transactions if t['amount'] > 0)
                    day_withdrawals = sum(abs(t['amount']) for t in day_transactions if t['amount'] < 0)
                    
                    total_daily_deposits += day_deposits
                    total_daily_withdrawals += day_withdrawals
                    
                    if day_deposits > 0 or day_withdrawals > 0:
                        active_users_count += 1
                    
                    # Check if this user contributed interest today
                    user_interest = sum(
                        entry['amount'] for entry in self.prize_pool.ledger 
                        if (entry['date'] == current_date and 
                            entry['source'] == f"user_{user.user_id}")
                    )
                    
                    if user_interest > 0.000001:  # Small threshold for floating point
                        users_with_interest += 1
                    
                    # Get balances for debugging
                    balance_start = user.get_balance(yesterday) if current_date > user.creation_date else 0
                    balance_end = user.get_balance(current_date)
                    total_balance_start += balance_start
                    total_balance_end += balance_end
            
            daily_record['total_deposits'] = total_daily_deposits
            daily_record['total_withdrawals'] = total_daily_withdrawals
            daily_record['users_active'] = active_users_count
            
            # Add debug information
            daily_record['debug_info'] = {
                'users_with_interest': users_with_interest,
                'total_balance_start': total_balance_start,
                'total_balance_end': total_balance_end,
                'users_created_today': sum(1 for user in self.users if user.creation_date == current_date),
                'users_eligible_for_interest': sum(1 for user in self.users if user.creation_date < current_date)
            }
            
            # Show details for first 5 users who had activity
            user_details_count = 0
            for user in self.users:
                if user_details_count >= 5:
                    break
                    
                if current_date >= user.creation_date:
                    day_transactions = [t for t in user.transactions if t['date'] == current_date]
                    day_deposits = sum(t['amount'] for t in day_transactions if t['amount'] > 0)
                    day_withdrawals = sum(abs(t['amount']) for t in day_transactions if t['amount'] < 0)
                    
                    user_interest = sum(
                        entry['amount'] for entry in self.prize_pool.ledger 
                        if (entry['date'] == current_date and 
                            entry['source'] == f"user_{user.user_id}")
                    )
                    
                    balance_start = user.get_balance(yesterday) if current_date > user.creation_date else 0
                    balance_end = user.get_balance(current_date)
                    
                    # Show user if they had ANY activity (deposits, withdrawals, or interest)
                    if (day_deposits > 0 or day_withdrawals > 0 or user_interest > 0.000001 or 
                        user.creation_date == current_date):
                        
                        user_record = {
                            'user_id': user.user_id,
                            'creation_date': user.creation_date,
                            'balance_start': balance_start,
                            'deposits': day_deposits,
                            'withdrawals': day_withdrawals,
                            'interest_contribution': user_interest,
                            'balance_end': balance_end,
                            'expected_change': day_deposits - day_withdrawals,
                            'created_today': user.creation_date == current_date
                        }
                        
                        daily_record['user_details'].append(user_record)
                        user_details_count += 1
            
            daily_data.append(daily_record)
        
        return daily_data

    def display_daily_extract(self, num_days=10):
        """Display detailed daily extract for validation"""
        st.subheader(f"📅 Daily Extract - First {num_days} Days")
        st.write("This shows exactly what happens each day to validate calculations")
        
        daily_data = self.generate_daily_extract(num_days)
        
        for day_record in daily_data:
            with st.expander(f"Day {day_record['day_number']} - {day_record['date']} (Click to expand)", expanded=(day_record['day_number'] <= 3)):
                
                # Daily summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prize Pool Start", f"€{day_record['pool_balance_start']:.6f}")
                    st.metric("Pool Interest Earned", f"€{day_record['pool_interest_earned']:.6f}")
                with col2:
                    st.metric("Total User Contributions", f"€{day_record['total_user_contributions']:.6f}")
                    st.metric("Users with Interest", day_record['debug_info']['users_with_interest'])
                with col3:
                    st.metric("Total Deposits (ALL users)", f"€{day_record['total_deposits']:.2f}")
                    st.metric("Active Users (transactions)", day_record['users_active'])
                
                st.metric("🏆 Prize Pool End", f"€{day_record['pool_balance_end']:.6f}", 
                         f"+€{day_record['pool_balance_end'] - day_record['pool_balance_start']:.6f}")
                
                # Debug information
                st.write("**Debug Info:**")
                debug = day_record['debug_info']
                st.write(f"- Users created today: {debug['users_created_today']}")
                st.write(f"- Users eligible for interest: {debug['users_eligible_for_interest']}")
                st.write(f"- Total balance start (all users): €{debug['total_balance_start']:.2f}")
                st.write(f"- Total balance end (all users): €{debug['total_balance_end']:.2f}")
                
                # Expected interest calculation
                if debug['users_eligible_for_interest'] > 0 and debug['total_balance_start'] > 0:
                    expected_interest = debug['total_balance_start'] * self.user_daily_rate
                    st.write(f"- Expected user interest: €{debug['total_balance_start']:.2f} × {self.user_daily_rate:.8f} = €{expected_interest:.6f}")
                    
                    if abs(expected_interest - day_record['total_user_contributions']) < 0.000001:
                        st.success(f"✅ Interest calculation matches!")
                    else:
                        st.error(f"❌ Interest mismatch: Expected €{expected_interest:.6f}, Got €{day_record['total_user_contributions']:.6f}")
                
                # User details
                if day_record['user_details']:
                    st.write("**User Activity (First 5 with activity):**")
                    
                    user_df_data = []
                    for user_record in day_record['user_details']:
                        created_marker = " 🆕" if user_record['created_today'] else ""
                        balance_match = abs((user_record['balance_start'] + user_record['expected_change']) - user_record['balance_end']) < 0.01
                        
                        user_df_data.append({
                            'User ID': user_record['user_id'] + created_marker,
                            'Created': user_record['creation_date'].strftime('%m-%d'),
                            'Balance Start': f"€{user_record['balance_start']:.2f}",
                            'Deposits': f"€{user_record['deposits']:.2f}" if user_record['deposits'] > 0 else "-",
                            'Withdrawals': f"€{user_record['withdrawals']:.2f}" if user_record['withdrawals'] > 0 else "-",
                            'Balance End': f"€{user_record['balance_end']:.2f}",
                            'Interest Contrib': f"€{user_record['interest_contribution']:.6f}",
                            'Balance ✓': "✅" if balance_match else "❌"
                        })
                    
                    if user_df_data:
                        user_df = pd.DataFrame(user_df_data)
                        st.dataframe(user_df, use_container_width=True)
                    else:
                        st.write("No significant user activity today")
                
                # Verification
                st.write("**Overall Verification:**")
                pool_growth = day_record['pool_balance_end'] - day_record['pool_balance_start']
                expected_growth = day_record['pool_interest_earned'] + day_record['total_user_contributions']
                
                if abs(pool_growth - expected_growth) < 0.000001:
                    st.success(f"✅ Pool growth matches: €{pool_growth:.6f} = €{day_record['pool_interest_earned']:.6f} (pool) + €{day_record['total_user_contributions']:.6f} (users)")
                else:
                    st.error(f"❌ Pool growth mismatch: Expected €{expected_growth:.6f}, Got €{pool_growth:.6f}")

        # Summary table
        st.subheader("📊 Daily Summary Table")
        summary_data = []
        for day_record in daily_data:
            summary_data.append({
                'Day': day_record['day_number'],
                'Date': day_record['date'].strftime('%Y-%m-%d'),
                'Pool Start': f"€{day_record['pool_balance_start']:.6f}",
                'Pool Interest': f"€{day_record['pool_interest_earned']:.6f}",
                'User Contributions': f"€{day_record['total_user_contributions']:.6f}",
                'Pool End': f"€{day_record['pool_balance_end']:.6f}",
                'Total Deposits': f"€{day_record['total_deposits']:.2f}",
                'Total Withdrawals': f"€{day_record['total_withdrawals']:.2f}",
                'Active Users': day_record['users_active']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

def create_streamlit_app():
    st.set_page_config(page_title="Prize-Linked Savings Simulator", layout="wide")
    
    st.title("🏆 Prize-Linked Savings Account Simulator")
    st.markdown("Simulate thousands of users to understand prize pool sustainability and business metrics.")
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    # User Population
    st.sidebar.subheader("👥 User Population")
    num_users = st.sidebar.slider("Number of Users", 100, 5000, 1000, step=100)
    simulation_days = st.sidebar.slider("Simulation Period (days)", 90, 730, 365, step=30)
    
    # Interest Rates
    st.sidebar.subheader("💰 Interest Rates")
    user_rate = st.sidebar.slider("User Interest Rate (APY)", 0.01, 0.10, 0.04, step=0.005)
    pool_rate = st.sidebar.slider("Prize Pool Interest Rate (APY)", 0.01, 0.10, 0.04, step=0.005)
    
    # Goal Parameters
    st.sidebar.subheader("🎯 Goal Settings")
    min_goal = st.sidebar.number_input("Minimum Goal (€)", 10, 200, 50)
    max_goal = st.sidebar.number_input("Maximum Goal (€)", 200, 1000, 500)
    
    # User Behavior
    st.sidebar.subheader("📊 User Behavior")
    deposit_prob = st.sidebar.slider("Monthly Deposit Probability", 0.3, 0.9, 0.7, step=0.05)
    withdrawal_prob = st.sidebar.slider("Monthly Withdrawal Probability", 0.05, 0.3, 0.15, step=0.05)
    goal_change_prob = st.sidebar.slider("Monthly Goal Change Probability", 0.01, 0.1, 0.05, step=0.01)
    
    # Transaction Ranges
    st.sidebar.subheader("💵 Transaction Ranges")
    initial_deposit_min = st.sidebar.number_input("Initial Deposit Min (€)", 100, 2000, 500)
    initial_deposit_max = st.sidebar.number_input("Initial Deposit Max (€)", 1000, 10000, 5000)
    monthly_deposit_min = st.sidebar.number_input("Monthly Deposit Min (€)", 10, 100, 50)
    monthly_deposit_max = st.sidebar.number_input("Monthly Deposit Max (€)", 100, 1000, 300)
    withdrawal_pct_min = st.sidebar.slider("Min Withdrawal % of Balance", 0.05, 0.3, 0.1, step=0.05)
    withdrawal_pct_max = st.sidebar.slider("Max Withdrawal % of Balance", 0.2, 0.8, 0.5, step=0.05)
    
    # Prize Settings
    st.sidebar.subheader("🏆 Prize Settings")
    monthly_multiplier = st.sidebar.slider("Monthly Prize Multiplier", 0.05, 0.25, 0.12, step=0.01)
    quarterly_multiplier = st.sidebar.slider("Quarterly Prize Multiplier", 0.10, 0.30, 0.15, step=0.01)
    
    # User Behavior Controls - Unified Profile System
    st.sidebar.subheader("👤 Unified User Profiles")
    st.sidebar.write("Each profile combines saving frequency and transaction amounts")

    st.sidebar.write("**Conservative Savers (Cautious with money):**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        conservative_deposit_freq = st.slider("Conservative: Monthly Deposit %", 0.1, 0.7, 0.3, step=0.05, key="conservative_deposit")
        conservative_withdrawal_freq = st.slider("Conservative: Monthly Withdrawal %", 0.05, 0.4, 0.2, step=0.05, key="conservative_withdraw")
        conservative_goal_change_freq = st.slider("Conservative: Goal Change %", 0.01, 0.15, 0.08, step=0.01, key="conservative_goal")
    with col2:
        conservative_initial_min = st.number_input("Conservative: Initial Min (€)", 50, 1000, 200, key="conservative_init_min")
        conservative_initial_max = st.number_input("Conservative: Initial Max (€)", 300, 2000, 800, key="conservative_init_max")
        conservative_monthly_min = st.number_input("Conservative: Monthly Min (€)", 10, 100, 30, key="conservative_month_min")
        conservative_monthly_max = st.number_input("Conservative: Monthly Max (€)", 50, 300, 120, key="conservative_month_max")

    st.sidebar.write("**Average Savers (Typical saving behavior):**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        average_deposit_freq = st.slider("Average: Monthly Deposit %", 0.2, 0.8, 0.5, step=0.05, key="average_deposit")
        average_withdrawal_freq = st.slider("Average: Monthly Withdrawal %", 0.1, 0.5, 0.25, step=0.05, key="average_withdraw")
        average_goal_change_freq = st.slider("Average: Goal Change %", 0.01, 0.2, 0.06, step=0.01, key="average_goal")
    with col2:
        average_initial_min = st.number_input("Average: Initial Min (€)", 200, 2000, 500, key="average_init_min")
        average_initial_max = st.number_input("Average: Initial Max (€)", 800, 4000, 2000, key="average_init_max")
        average_monthly_min = st.number_input("Average: Monthly Min (€)", 50, 200, 80, key="average_month_min")
        average_monthly_max = st.number_input("Average: Monthly Max (€)", 150, 600, 350, key="average_month_max")

    st.sidebar.write("**Aggressive Savers (High income, ambitious goals):**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        aggressive_deposit_freq = st.slider("Aggressive: Monthly Deposit %", 0.4, 0.9, 0.7, step=0.05, key="aggressive_deposit")
        aggressive_withdrawal_freq = st.slider("Aggressive: Monthly Withdrawal %", 0.05, 0.3, 0.15, step=0.05, key="aggressive_withdraw")
        aggressive_goal_change_freq = st.slider("Aggressive: Goal Change %", 0.01, 0.15, 0.04, step=0.01, key="aggressive_goal")
    with col2:
        aggressive_initial_min = st.number_input("Aggressive: Initial Min (€)", 500, 3000, 1000, key="aggressive_init_min")
        aggressive_initial_max = st.number_input("Aggressive: Initial Max (€)", 2000, 8000, 5000, key="aggressive_init_max")
        aggressive_monthly_min = st.number_input("Aggressive: Monthly Min (€)", 150, 500, 250, key="aggressive_month_min")
        aggressive_monthly_max = st.number_input("Aggressive: Monthly Max (€)", 400, 1200, 700, key="aggressive_month_max")

    # Profile Distribution
    st.sidebar.subheader("📊 Profile Distribution")
    conservative_pct = st.sidebar.slider("Conservative Savers %", 0.1, 0.6, 0.3, step=0.05)
    aggressive_pct = st.sidebar.slider("Aggressive Savers %", 0.1, 0.4, 0.2, step=0.05)
    # Average percentage is calculated automatically
    average_pct = max(0.1, 1.0 - conservative_pct - aggressive_pct)
    st.sidebar.write(f"**Average Savers: {average_pct:.1%}** (calculated automatically)")
    
    # User Growth & Churn
    st.sidebar.subheader("📈 User Growth & Churn")
    st.sidebar.write("Control how users join and leave over time")
    monthly_growth_rate = st.sidebar.slider("Monthly Growth Rate", 0.0, 0.20, 0.05, step=0.01, 
                                           help="Percentage of current users to add each month")
    monthly_churn_rate = st.sidebar.slider("Monthly Churn Rate", 0.0, 0.10, 0.02, step=0.005,
                                          help="Percentage of current users to churn each month")
    
    # Create simulation parameters
    params = SimulationParams(
        num_users=num_users,
        simulation_days=simulation_days,
        user_interest_rate=user_rate,
        pool_interest_rate=pool_rate,
        min_goal=min_goal,
        max_goal=max_goal,
        monthly_deposit_probability=deposit_prob,
        withdrawal_probability=withdrawal_prob,
        goal_change_probability=goal_change_prob,
        initial_deposit_min=initial_deposit_min,
        initial_deposit_max=initial_deposit_max,
        monthly_deposit_min=monthly_deposit_min,
        monthly_deposit_max=monthly_deposit_max,
        withdrawal_percentage_min=withdrawal_pct_min,
        withdrawal_percentage_max=withdrawal_pct_max,
        monthly_prize_multiplier=monthly_multiplier,
        quarterly_prize_multiplier=quarterly_multiplier,
        monthly_growth_rate=monthly_growth_rate,
        monthly_churn_rate=monthly_churn_rate
    )
    
    # Run Simulation Button
    if st.button("🚀 Run Simulation", type="primary"):
        # Collect unified behavior parameters
        unified_behavior_params = {
            'conservative_pct': conservative_pct,
            'aggressive_pct': aggressive_pct,
            'conservative_deposit_freq': conservative_deposit_freq,
            'conservative_withdrawal_freq': conservative_withdrawal_freq,
            'conservative_goal_change_freq': conservative_goal_change_freq,
            'conservative_initial_min': conservative_initial_min,
            'conservative_initial_max': conservative_initial_max,
            'conservative_monthly_min': conservative_monthly_min,
            'conservative_monthly_max': conservative_monthly_max,
            'average_deposit_freq': average_deposit_freq,
            'average_withdrawal_freq': average_withdrawal_freq,
            'average_goal_change_freq': average_goal_change_freq,
            'average_initial_min': average_initial_min,
            'average_initial_max': average_initial_max,
            'average_monthly_min': average_monthly_min,
            'average_monthly_max': average_monthly_max,
            'aggressive_deposit_freq': aggressive_deposit_freq,
            'aggressive_withdrawal_freq': aggressive_withdrawal_freq,
            'aggressive_goal_change_freq': aggressive_goal_change_freq,
            'aggressive_initial_min': aggressive_initial_min,
            'aggressive_initial_max': aggressive_initial_max,
            'aggressive_monthly_min': aggressive_monthly_min,
            'aggressive_monthly_max': aggressive_monthly_max,
        }
        
        # Store results in session state
        simulator = MultiUserSimulator(params)
        
        with st.spinner("Running simulation..."):
            simulator.run_simulation(unified_behavior_params)
            basic_metrics = simulator.calculate_metrics()
            
            # Store in session state
            st.session_state.simulation_complete = True
            st.session_state.simulator = simulator
            st.session_state.basic_metrics = basic_metrics
            st.session_state.params = params
        
        st.success("Simulation complete!")
    
    # Display results if simulation has been run
    if st.session_state.get('simulation_complete', False):
        simulator = st.session_state.simulator
        basic_metrics = st.session_state.basic_metrics
        
        # Add detailed analytics with error handling and unique session management
        try:
            import time
            import random
            
            # Create unique keys with timestamp and random component
            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            unique_prefix = f"{timestamp}_{random_suffix}"
            
            st.write("🔍 Generating detailed analytics...")
            detailed_metrics = simulator.calculate_detailed_metrics()
            st.write("✅ Detailed metrics calculated successfully")
            
            # Pass the unique prefix to the display method
            simulator.display_detailed_analytics(detailed_metrics, unique_prefix)
            st.write("✅ Detailed analytics displayed successfully")
        except Exception as e:
            st.error(f"❌ Error in detailed analytics: {str(e)}")
            st.write("Debug info:")
            st.write(f"Number of users: {len(simulator.users)}")
            st.write(f"Users have behavior profiles: {hasattr(simulator.users[0], 'behavior_profile') if simulator.users else 'No users'}")
            import traceback
            st.code(traceback.format_exc())
        
        # Display basic results
        st.header("📈 Simulation Results")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", f"{basic_metrics['total_users']:,}")
            st.metric("Avg Balance/User", f"€{basic_metrics['avg_balance_per_user']:,.2f}")
        with col2:
            st.metric("Total Balance", f"€{basic_metrics['total_balance']:,.2f}")
            st.metric("Prize Pool", f"€{basic_metrics['prize_pool_balance']:,.2f}")
        with col3:
            pool_ratio = basic_metrics['prize_pool_balance'] / max(basic_metrics['total_balance'], 1) * 100
            st.metric("Pool/Balance Ratio", f"{pool_ratio:.2f}%")
            st.metric("Avg Cycles/User", f"{basic_metrics['avg_cycles_completed']:.1f}")
        with col4:
            st.metric("Interest Contributed", f"€{basic_metrics['total_interest_contributed']:,.2f}")
            st.metric("Pool Interest Earned", f"€{basic_metrics['pool_interest_earned']:,.2f}")
        
        # User Distribution Chart
        st.subheader("👥 User Behavior Distribution")
        behavior_data = pd.DataFrame({
            'Profile': ['Conservative Savers', 'Average Savers', 'Aggressive Savers'],
            'Count': [basic_metrics['conservative_savers'], basic_metrics['average_savers'], basic_metrics['aggressive_savers']],
            'Percentage': [
                basic_metrics['conservative_savers'] / basic_metrics['total_users'] * 100,
                basic_metrics['average_savers'] / basic_metrics['total_users'] * 100,
                basic_metrics['aggressive_savers'] / basic_metrics['total_users'] * 100
            ]
        })
        
        fig_behavior = px.pie(
            behavior_data, 
            values='Count', 
            names='Profile',
            title="User Behavior Distribution",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        st.plotly_chart(fig_behavior, use_container_width=True, key=f"main_user_behavior_pie_{timestamp}_{random_suffix}")
        
        # Prize Analysis
        st.subheader("💰 Prize Pool Analysis")
        
        # Calculate users eligible for prizes (those who achieved goals)
        users_eligible_for_prizes = sum(1 for user in simulator.users if user.cumulative_achieved_goal_amount > 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Prize Pool", f"€{basic_metrics['prize_pool_balance']:,.2f}")
            st.metric("Users Eligible for Prizes", f"{users_eligible_for_prizes:,}")
        
        with col2:
            # Calculate monthly prize pool growth
            if st.session_state.params.simulation_days >= 30:
                monthly_pool_growth = (basic_metrics['total_interest_contributed'] + basic_metrics['pool_interest_earned']) / (st.session_state.params.simulation_days / 30)
                st.metric("Monthly Pool Growth", f"€{monthly_pool_growth:,.2f}")
            else:
                st.metric("Monthly Pool Growth", "N/A (Sim < 30 days)")
            
            st.metric("Eligibility Rate", f"{users_eligible_for_prizes / basic_metrics['total_users']:.1%}")
        
        with col3:
            st.metric("User Interest Contributed", f"€{basic_metrics['total_interest_contributed']:,.2f}")
            st.metric("Pool Interest Earned", f"€{basic_metrics['pool_interest_earned']:,.2f}")
        
        # Sustainability Analysis
        st.subheader("📊 Sustainability Analysis")
        
        # Calculate annual growth rates
        daily_pool_growth = (basic_metrics['pool_interest_earned'] + basic_metrics['total_interest_contributed']) / st.session_state.params.simulation_days
        annual_pool_growth = daily_pool_growth * 365
        
        # Estimate user balance growth (assume 10% annually from deposits)
        estimated_annual_balance_growth = basic_metrics['total_balance'] * 0.1
        
        sustainability_ratio = annual_pool_growth / max(estimated_annual_balance_growth, 1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Daily Pool Growth", f"€{daily_pool_growth:.2f}")
        with col2:
            st.metric("Estimated Annual Pool Growth", f"€{annual_pool_growth:,.2f}")
        with col3:
            st.metric("Sustainability Ratio", f"{sustainability_ratio:.2f}")
        
        if sustainability_ratio > 1.2:
            st.success("✅ **Highly Sustainable**: Prize pool is growing faster than user balances!")
        elif sustainability_ratio > 0.8:
            st.warning("⚠️ **Moderately Sustainable**: Prize pool growth is keeping pace with user growth.")
        else:
            st.error("❌ **Potentially Unsustainable**: Prize pool may not keep up with user growth.")
        
        # Add monthly goal analysis
        simulator.display_monthly_goal_analysis()

        # Add monthly financial summary
        simulator.display_monthly_financial_summary()

        # Add monthly lifecycle summary
        simulator.display_monthly_lifecycle_summary()

        # Add daily extract for validation - NOW THIS WILL WORK!
        if st.checkbox("🔍 Show Daily Extract (for validation)", value=False):
            num_days = st.slider("Number of days to show", 5, 30, 10)
            simulator.display_daily_extract(num_days)
        
        # Export functionality
        st.subheader("📥 Export Results")
        results_df = pd.DataFrame([basic_metrics])
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"simulation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    create_streamlit_app()
