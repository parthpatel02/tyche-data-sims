import streamlit as st
import pandas as pd
import numpy as np
import datetime
import uuid
import math

# --- App Configuration ---
st.set_page_config(
    page_title="Investment & Prize Simulator",
    page_icon="💼",
    layout="wide"
)

# --- Pre-loaded Investment Data ---
INVESTMENT_OPTIONS = {
    "iShares Core U.S. Aggregate Bond ETF (AGG)": {"return": 0.047, "type": "Bond"},
    "Vanguard Total Bond Market ETF (BND)": {"return": 0.045, "type": "Bond"},
    "iShares 1-3 Year Treasury Bond ETF (SHY)": {"return": 0.032, "type": "Bond"},
    "SPDR Bloomberg Convertible Securities ETF (CWB)": {"return": 0.149, "type": "Bond"},
    "Vanguard S&P 500 ETF (VOO)": {"return": 0.235, "type": "Equity"},
    "Invesco QQQ Trust (QQQ)": {"return": 0.290, "type": "Equity"},
    "iShares Russell 2000 ETF (IWM)": {"return": 0.150, "type": "Equity"},
    "VanEck Gold Miners ETF (GDX)": {"return": 0.546, "type": "Equity"},
    "Global X MSCI Greece ETF (GREK)": {"return": 0.579, "type": "Equity"},
    "iShares MSCI Europe Financials ETF (EUFN)": {"return": 0.397, "type": "Equity"},
    "ARK Next Generation Internet ETF (ARKW)": {"return": 0.390, "type": "Equity"},
    "Global X Uranium ETF (URA)": {"return": 0.427, "type": "Commodity"},
    "United States Oil Fund (USO)": {"return": 0.050, "type": "Commodity"},
    "Invesco DB Commodity Index Tracking Fund (DBC)": {"return": 0.075, "type": "Commodity"},
    "abrdn Physical Platinum Shares ETF (PPLT)": {"return": 0.445, "type": "Commodity"},
    "iShares MSCI Brazil ETF (EWZ)": {"return": 0.220, "type": "Emerging Market"},
    "iShares MSCI India ETF (INDA)": {"return": 0.180, "type": "Emerging Market"},
    "VanEck Vietnam ETF (VNM)": {"return": 0.389, "type": "Emerging Market"},
    "iShares MSCI Turkey ETF (TUR)": {"return": 0.450, "type": "Emerging Market"},
    "Global X MSCI Argentina ETF (ARGT)": {"return": 0.635, "type": "Emerging Market"},
}

# --- Helper Functions ---
def get_draw_type(current_date):
    if current_date.day == 24: return "Monthly"
    if current_date.weekday() == 5: return "Weekly"
    return None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def get_end_of_month(date):
    next_month = date.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)

def run_simulation(config):
    # (Data structure initialization as before)
    persons_data, users_data, transactions_data, ledger_entries_data = [], [], [], []
    draws_data, qualification_tracker_data, user_goals_data = [], [], []
    user_interests_data, user_acc_details_data = [], []
    # NEW: Added interest log
    daily_interest_log_data = []
    goals_achieved_last_month, goals_failed_last_month = 0, 0
    
    prizes_this_month = config['base_winners_per_draw']

    # --- Initial User Population ---
    for i in range(config["initial_users"]):
        person_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
        creation_date = datetime.date(2024, 1, 1)
        persons_data.append({"person_id": person_id, "name": f"Person {i}", "status": "ACTIVE", "country_residence": "NL", "date_created": creation_date})
        users_data.append({"user_id": user_id, "person_id": person_id, "status": "ACTIVE", "persona": np.random.choice(list(config["user_personas"].keys()), p=list(config["user_personas"].values())), "date_created": creation_date})
        user_acc_details_data.append({"id": str(uuid.uuid4()), "user_id": user_id, "bank_reference": f"bank_ref_{i}", "date_created": creation_date})
        user_interests_data.append({"interests_selection_id": str(uuid.uuid4()), "user_id": user_id, "interest_type": "cash_prize", "date_created": creation_date})

    # --- Simulation Loop ---
    start_date = datetime.date(2024, 1, 1)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for day_num in range(config["simulation_days"]):
        current_date = start_date + datetime.timedelta(days=day_num)
        
        # --- Daily Interest Calculation ---
        ledger_df_temp = pd.DataFrame(ledger_entries_data)
        total_deposits = 0
        if not ledger_df_temp.empty:
            user_balances = ledger_df_temp.groupby('account_id')['amount'].sum()
            total_deposits = user_balances[user_balances.index.str.startswith('user_', na=False)].sum()
        
        low_risk_pool = total_deposits * (config["low_risk_allocation"] / 100)
        high_risk_pool = total_deposits * (1 - (config["low_risk_allocation"] / 100))
        low_risk_return = low_risk_pool * (config["low_risk_apy"] / 365)
        high_risk_return = high_risk_pool * (config["high_risk_return"] / 365)
        interest_earned = low_risk_return + high_risk_return
        
        # NEW: Log interest breakdown
        daily_interest_log_data.append({"date": current_date, "low_risk_return": low_risk_return, "high_risk_return": high_risk_return})

        if interest_earned > 0:
            tx_id = str(uuid.uuid4())
            prize_pool_share = interest_earned * config["prize_pool_percentage"]
            revenue_share = interest_earned * (1 - config["prize_pool_percentage"])
            transactions_data.append({"transaction_uuid": tx_id, "type": "INTEREST_ACCRUAL", "amount": interest_earned, "status": "COMPLETED", "date_created": current_date, "user_id": "platform"})
            ledger_entries_data.extend([
                {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": "INTEREST_EXPENSE", "ledger_entry_type": "debit", "amount": -interest_earned, "date_created": current_date},
                {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": "PRIZE_POOL", "ledger_entry_type": "credit", "amount": prize_pool_share, "date_created": current_date},
                {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": "PLATFORM_REVENUE", "ledger_entry_type": "credit", "amount": revenue_share, "date_created": current_date}
            ])

        # --- Monthly Admin ---
        if current_date.day == 1 and day_num > 0:
            last_month_date = current_date - datetime.timedelta(days=1)
            
            # 1. Evaluate previous month's goals
            goals_achieved_last_month, goals_failed_last_month = 0, 0
            if user_goals_data:
                prize_pool_balance = ledger_df_temp.query("account_id == 'PRIZE_POOL'")['amount'].sum() if not ledger_df_temp.empty and "PRIZE_POOL" in ledger_df_temp['account_id'].values else 0
                for goal in user_goals_data:
                    if goal['status'] == 'ACTIVE' and goal['date_end'] == last_month_date:
                        start_of_goal_month = goal['date_end'].replace(day=1)
                        user_deposits = ledger_df_temp[
                            (ledger_df_temp['account_id'] == f"user_{goal['user_id']}") &
                            (pd.to_datetime(ledger_df_temp['date_created']).dt.date >= start_of_goal_month) &
                            (pd.to_datetime(ledger_df_temp['date_created']).dt.date <= goal['date_end']) &
                            (ledger_df_temp['ledger_entry_type'] == 'credit')
                        ]['amount'].sum() if not ledger_df_temp.empty else 0
                        if user_deposits >= goal['target_amount']:
                            goal['status'] = 'ACHIEVED'
                            goals_achieved_last_month += 1
                            reward_amount = config['guaranteed_reward_amount']
                            if prize_pool_balance >= reward_amount:
                                tx_id = str(uuid.uuid4())
                                transactions_data.append({"transaction_uuid": tx_id, "type": "GOAL_REWARD", "amount": reward_amount, "status": "COMPLETED", "date_created": current_date, "user_id": goal["user_id"]})
                                ledger_entries_data.extend([
                                    {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": "PRIZE_POOL", "ledger_entry_type": "debit", "amount": -reward_amount, "date_created": current_date},
                                    {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": f"user_{goal['user_id']}", "ledger_entry_type": "credit", "amount": reward_amount, "date_created": current_date}
                                ])
                                prize_pool_balance -= reward_amount
                        else:
                            goal['status'] = 'FAILED'
                            goals_failed_last_month += 1
            
            bonus_winners = math.floor(goals_achieved_last_month / config['bonus_winner_threshold'])
            prizes_this_month = config['base_winners_per_draw'] + bonus_winners

            # 2. User Growth & Churn
            active_users = [u for u in users_data if u["status"] == "ACTIVE"]
            new_users_count = int(len(active_users) * config["monthly_user_growth_rate"])
            for _ in range(new_users_count):
                person_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
                persons_data.append({"person_id": person_id, "name": f"Person {len(persons_data)}", "status": "ACTIVE", "country_residence": "NL", "date_created": current_date})
                users_data.append({"user_id": user_id, "person_id": person_id, "status": "ACTIVE", "persona": np.random.choice(list(config["user_personas"].keys()), p=list(config["user_personas"].values())), "date_created": current_date})
                user_acc_details_data.append({"id": str(uuid.uuid4()), "user_id": user_id, "bank_reference": f"bank_ref_{len(persons_data)}", "date_created": current_date})
                user_interests_data.append({"interests_selection_id": str(uuid.uuid4()), "user_id": user_id, "interest_type": "cash_prize", "date_created": current_date})
            for user in active_users:
                if np.random.rand() < config["monthly_churn_rate"]: user["status"] = "INACTIVE"
            
            # 3. New Goal Setting
            active_users = [u for u in users_data if u["status"] == "ACTIVE"]
            for user in active_users:
                if np.random.rand() < config["monthly_goal_setting_prob"]:
                    target_amount = np.random.uniform(config["goal_amount_min"], config["goal_amount_max"])
                    user_goals_data.append({"monthly_goal_id": str(uuid.uuid4()), "user_id": user["user_id"], "target_amount": target_amount, "date_created": current_date, "date_end": get_end_of_month(current_date), "status": "ACTIVE"})

        # --- User Activity (Daily) ---
        for user in users_data:
            if user["status"] == "ACTIVE":
                persona_config = config[f"persona_{user['persona']}"]
                # Deposit Logic
                has_active_goal = any(g['user_id'] == user['user_id'] and g['status'] == 'ACTIVE' for g in user_goals_data)
                deposit_prob = persona_config["daily_deposit_probability"] * (config["goal_motivation_multiplier"] if has_active_goal else 1.0)
                if np.random.rand() < deposit_prob:
                    deposit_amount = np.random.uniform(persona_config["deposit_amount_min"], persona_config["deposit_amount_max"])
                    tx_id = str(uuid.uuid4())
                    transactions_data.append({"transaction_uuid": tx_id, "type": "DEPOSIT", "amount": deposit_amount, "status": "COMPLETED", "date_created": current_date, "user_id": user["user_id"]})
                    ledger_entries_data.extend([
                        {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": f"user_{user['user_id']}", "ledger_entry_type": "credit", "amount": deposit_amount, "date_created": current_date},
                        {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": "USER_SAVINGS_POOL", "ledger_entry_type": "debit", "amount": -deposit_amount, "date_created": current_date}
                    ])
                
                # NEW: Withdrawal Logic
                if np.random.rand() < persona_config["daily_withdrawal_probability"]:
                    user_balance = ledger_df_temp[ledger_df_temp['account_id'] == f"user_{user['user_id']}"]['amount'].sum() if not ledger_df_temp.empty else 0
                    if user_balance > 0:
                        withdrawal_amount = np.random.uniform(persona_config["withdrawal_amount_min"], min(user_balance, persona_config["withdrawal_amount_max"]))
                        tx_id = str(uuid.uuid4())
                        transactions_data.append({"transaction_uuid": tx_id, "type": "WITHDRAWAL", "amount": withdrawal_amount, "status": "COMPLETED", "date_created": current_date, "user_id": user["user_id"]})
                        ledger_entries_data.extend([
                            {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": f"user_{user['user_id']}", "ledger_entry_type": "debit", "amount": -withdrawal_amount, "date_created": current_date},
                            {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": "USER_SAVINGS_POOL", "ledger_entry_type": "credit", "amount": withdrawal_amount, "date_created": current_date}
                        ])

        # --- Draw Logic ---
        draw_type = get_draw_type(current_date)
        if draw_type:
            ledger_df_temp_draw = pd.DataFrame(ledger_entries_data)
            prize_pool_balance = 0
            if not ledger_df_temp_draw.empty and "PRIZE_POOL" in ledger_df_temp_draw['account_id'].values:
                prize_pool_balance = ledger_df_temp_draw.query("account_id == 'PRIZE_POOL'")['amount'].sum()
            if prize_pool_balance > 0:
                draw_id = str(uuid.uuid4())
                draws_data.append({"draw_id": draw_id, "draw_type": draw_type, "date_draw": current_date, "total_prize_pool": prize_pool_balance, "number_of_prizes": prizes_this_month, "prize_value": "Dynamic"})
                if not ledger_df_temp_draw.empty:
                    user_balances_df = ledger_df_temp_draw[ledger_df_temp_draw['account_id'].str.startswith('user_', na=False)].groupby('account_id')['amount'].sum().reset_index()
                    user_balances_df = user_balances_df.rename(columns={"amount": "balance"})
                    user_balances_df['user_id'] = user_balances_df['account_id'].str.replace('user_', '')
                    eligible_users_df = pd.merge(pd.DataFrame(users_data), user_balances_df, on="user_id")
                    eligible_users_df = eligible_users_df[eligible_users_df['balance'] > 0]
                    if not eligible_users_df.empty:
                        goals_df = pd.DataFrame(user_goals_data)
                        if not goals_df.empty:
                            last_month_end = current_date.replace(day=1) - datetime.timedelta(days=1)
                            achieved_goals = goals_df[(goals_df['status'] == 'ACHIEVED') & (pd.to_datetime(goals_df['date_end']).dt.date == last_month_end)]
                            eligible_users_df = pd.merge(eligible_users_df, achieved_goals[['user_id']], on='user_id', how='left', indicator=True)
                            eligible_users_df['goal_bonus'] = np.where(eligible_users_df['_merge'] == 'both', config['goal_achievement_bonus'], 1.0)
                            eligible_users_df.drop(columns=['_merge'], inplace=True)
                        else:
                            eligible_users_df['goal_bonus'] = 1.0
                        eligible_users_df['account_age_months'] = (pd.to_datetime(current_date) - pd.to_datetime(eligible_users_df['date_created'])).dt.days / 30.44
                        eligible_users_df['duration_factor'] = 1 + (eligible_users_df['account_age_months'] / 12)
                        eligible_users_df['weight'] = eligible_users_df['balance'] * eligible_users_df['duration_factor'] * eligible_users_df['goal_bonus']
                        num_prizes_to_award = min(prizes_this_month, len(eligible_users_df))
                        if num_prizes_to_award > 0 and eligible_users_df['weight'].sum() > 0:
                            winners_df = eligible_users_df.sample(n=num_prizes_to_award, weights='weight', replace=False)
                            for _, winner in winners_df.iterrows():
                                start_of_last_month = (current_date.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
                                end_of_last_month = current_date.replace(day=1) - datetime.timedelta(days=1)
                                winner_deposits_last_month = ledger_df_temp_draw[(ledger_df_temp_draw['account_id'] == f"user_{winner['user_id']}") & (pd.to_datetime(ledger_df_temp_draw['date_created']).dt.date >= start_of_last_month) & (pd.to_datetime(ledger_df_temp_draw['date_created']).dt.date <= end_of_last_month) & (ledger_df_temp_draw['ledger_entry_type'] == 'credit')]['amount'].sum()
                                prize_value = max(winner_deposits_last_month * config['prize_percentage_of_savings'], config['min_prize_value'])
                                if prize_pool_balance - prize_value < 0: continue
                                tx_id = str(uuid.uuid4())
                                transactions_data.append({"transaction_uuid": tx_id, "type": "PRIZE_WIN", "amount": prize_value, "status": "COMPLETED", "date_created": current_date, "user_id": winner["user_id"]})
                                ledger_entries_data.extend([{"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": "PRIZE_POOL", "ledger_entry_type": "debit", "amount": -prize_value, "date_created": current_date}, {"ledger_entry_id": str(uuid.uuid4()), "transaction_uuid": tx_id, "account_id": f"user_{winner['user_id']}", "ledger_entry_type": "credit", "amount": prize_value, "date_created": current_date}])
                                qualification_tracker_data.append({"qualification_id": str(uuid.uuid4()), "user_id": winner["user_id"], "draw_id": draw_id, "qualification_score": winner["weight"], "is_winner": True, "prize_amount_won": prize_value, "date_calculated": current_date})
                                prize_pool_balance -= prize_value

        progress_bar.progress((day_num + 1) / config["simulation_days"])
        status_text.text(f"Simulating... {current_date.strftime('%Y-%m-%d')}")

    status_text.text("Simulation Complete!")
    
    all_dfs = {
        "Persons": pd.DataFrame(persons_data), "Users": pd.DataFrame(users_data),
        "User_Account_Details": pd.DataFrame(user_acc_details_data), "User_Interests": pd.DataFrame(user_interests_data),
        "User_Goals": pd.DataFrame(user_goals_data), "Transactions": pd.DataFrame(transactions_data),
        "Ledger_Entries": pd.DataFrame(ledger_entries_data), "Draws": pd.DataFrame(draws_data),
        "Qualification_Tracker": pd.DataFrame(qualification_tracker_data),
        "Daily_Interest_Log": pd.DataFrame(daily_interest_log_data)
    }
    return all_dfs, goals_achieved_last_month, goals_failed_last_month, prizes_this_month

# --- Streamlit UI ---
st.title("💼 Investment & Prize Simulator")
st.markdown("Model financial outcomes based on a **dual-investment strategy** and **monthly user savings goals**.")

# --- Investment Strategy Section ---
st.header("1. Define Investment Strategy")
col1, col2 = st.columns(2)
with col1:
    low_risk_allocation = st.slider("Low-Risk Allocation (%)", 0, 100, 95)
    low_risk_apy = st.slider("Low-Risk APY (%)", 0.1, 10.0, 4.0) / 100
with col2:
    st.markdown(f"**High-Risk Allocation: {100 - low_risk_allocation}%**")
    instrument_options = list(INVESTMENT_OPTIONS.keys())
    selected_instrument_name = st.selectbox("Select High-Risk Instrument", instrument_options, format_func=lambda name: f"{name} ({INVESTMENT_OPTIONS[name]['return']*100:.2f}% 1Y)")
high_risk_return = INVESTMENT_OPTIONS[selected_instrument_name]["return"]

st.markdown("---")

# --- Sidebar for Simulation Parameters ---
with st.sidebar:
    st.header("⚙️ Simulation Configuration")
    simulation_months = st.slider("Simulation Duration (Months)", 1, 36, 12)
    initial_users = st.number_input("Initial Users", 10, 50000, 10000)
    
    with st.expander("Financial & Growth Model", expanded=True):
        prize_pool_percentage = st.slider("Interest to Prize Pool (%)", 50, 100, 95)
        monthly_user_growth_rate = st.slider("Monthly User Growth Rate (%)", 0.0, 20.0, 5.0) / 100
        monthly_churn_rate = st.slider("Monthly Churn Rate (%)", 0.0, 10.0, 1.0) / 100
    
    with st.expander("Prize & Goal Structure", expanded=True):
        base_winners_per_draw = st.number_input("Base Winners per Draw", 1, 50, 5)
        bonus_winner_threshold = st.number_input("Bonus Winner per X Achievers", 1, 100, 10)
        guaranteed_reward_amount = st.number_input("Guaranteed Reward for Achievers (€)", 0, 100, 1)
        prize_percentage_of_savings = st.slider("Prize as % of Monthly Savings", 1, 100, 10) / 100
        min_prize_value = st.number_input("Minimum Prize Value (€)", 1, 100, 5)
        goal_achievement_bonus = st.slider("Goal Achievement Bonus Multiplier", 1.0, 5.0, 2.0)
        
    with st.expander("User Goal Behavior", expanded=True):
        monthly_goal_setting_prob = st.slider("Monthly Goal Setting Prob (%)", 1, 100, 25) / 100
        goal_amount_min, goal_amount_max = st.slider("Monthly Goal Amount Range (€)", 50, 2000, (100, 500))
        goal_motivation_multiplier = st.slider("Goal Motivation Multiplier", 1.0, 10.0, 3.0)

    with st.expander("User Personas & Behavior", expanded=True):
        persona_cautious_percentage = st.slider("Cautious Saver (%)", 0, 100, 40)
        st.markdown("**Cautious Saver**")
        c_daily_deposit_prob = st.slider("Daily Deposit Prob (%)", 0.1, 10.0, 0.5, key="c_d_prob") / 100
        c_deposit_min, c_deposit_max = st.slider("Deposit Range (€)", 1, 250, (5, 25), key="c_d_range")
        c_daily_withdrawal_prob = st.slider("Daily Withdrawal Prob (%)", 0.1, 10.0, 0.2, key="c_w_prob") / 100
        c_withdrawal_min, c_withdrawal_max = st.slider("Withdrawal Range (€)", 1, 250, (10, 50), key="c_w_range")

        st.markdown("**Active Depositor**")
        a_daily_deposit_prob = st.slider("Daily Deposit Prob (%)", 0.1, 10.0, 2.0, key="a_d_prob") / 100
        a_deposit_min, a_deposit_max = st.slider("Deposit Range (€)", 1, 500, (20, 75), key="a_d_range")
        a_daily_withdrawal_prob = st.slider("Daily Withdrawal Prob (%)", 0.1, 10.0, 0.5, key="a_w_prob") / 100
        a_withdrawal_min, a_withdrawal_max = st.slider("Withdrawal Range (€)", 1, 500, (20, 100), key="a_w_range")

run_button = st.button("🚀 Run Simulation")

if 'results' not in st.session_state:
    st.session_state.results = None

if run_button:
    config = {
        "simulation_days": simulation_months * 30, "initial_users": initial_users,
        "low_risk_allocation": low_risk_allocation, "low_risk_apy": low_risk_apy,
        "high_risk_return": high_risk_return,
        "prize_pool_percentage": prize_pool_percentage / 100,
        "monthly_user_growth_rate": monthly_user_growth_rate, "monthly_churn_rate": monthly_churn_rate,
        "user_personas": {"cautious": persona_cautious_percentage / 100, "active": (100-persona_cautious_percentage) / 100},
        "persona_cautious": {"daily_deposit_probability": c_daily_deposit_prob, "deposit_amount_min": c_deposit_min, "deposit_amount_max": c_deposit_max, "daily_withdrawal_probability": c_daily_withdrawal_prob, "withdrawal_amount_min": c_withdrawal_min, "withdrawal_amount_max": c_withdrawal_max},
        "persona_active": {"daily_deposit_probability": a_daily_deposit_prob, "deposit_amount_min": a_deposit_min, "deposit_amount_max": a_deposit_max, "daily_withdrawal_probability": a_daily_withdrawal_prob, "withdrawal_amount_min": a_withdrawal_min, "withdrawal_amount_max": a_withdrawal_max},
        "base_winners_per_draw": base_winners_per_draw, "bonus_winner_threshold": bonus_winner_threshold,
        "guaranteed_reward_amount": guaranteed_reward_amount,
        "prize_percentage_of_savings": prize_percentage_of_savings,
        "min_prize_value": min_prize_value, "goal_achievement_bonus": goal_achievement_bonus,
        "monthly_goal_setting_prob": monthly_goal_setting_prob,
        "goal_amount_min": goal_amount_min, "goal_amount_max": goal_amount_max,
        "goal_motivation_multiplier": goal_motivation_multiplier,
    }
    
    all_dfs, achieved, failed, prizes_this_month = run_simulation(config)
    st.session_state.results = {
        "all_dfs": all_dfs, "achieved": achieved, "failed": failed,
        "config": config, "prizes_this_month": prizes_this_month
    }

if st.session_state.results:
    results_data = st.session_state.results
    if isinstance(results_data, dict):
        all_dfs = results_data.get("all_dfs")
        achieved = results_data.get("achieved")
        failed = results_data.get("failed")
        config = results_data.get("config")
        prizes_this_month = results_data.get("prizes_this_month")
    
        st.header("📈 Simulation Results")
        
        # --- Calculate All KPIs ---
        ledger_df = all_dfs["Ledger_Entries"]
        transactions_df = all_dfs["Transactions"]
        users_df = all_dfs["Users"]
        goals_df = all_dfs["User_Goals"]
        qual_df = all_dfs["Qualification_Tracker"]

        # --- Financial KPIs ---
        final_revenue = ledger_df.query("account_id == 'PLATFORM_REVENUE'")['amount'].sum() if not ledger_df.empty else 0
        total_prizes = ledger_df.query("account_id == 'PRIZE_POOL' and ledger_entry_type == 'debit'")['amount'].sum() * -1 if not ledger_df.empty else 0
        total_interest = final_revenue + total_prizes
        final_deposits = ledger_df[ledger_df['account_id'].str.startswith('user_')]['amount'].sum() if not ledger_df.empty else 0
        avg_user_balance = final_deposits / users_df['user_id'].nunique() if not users_df.empty and users_df['user_id'].nunique() > 0 else 0
        
        # --- User & Engagement KPIs ---
        depositing_users = transactions_df[transactions_df['type'] == 'DEPOSIT']['user_id'].nunique() if not transactions_df.empty else 0
        active_users_count = users_df[users_df['status'] == 'ACTIVE']['user_id'].nunique() if not users_df.empty else 0
        goal_setters = goals_df['user_id'].nunique() if not goals_df.empty else 0
        engagement_rate = (goal_setters / active_users_count) * 100 if active_users_count > 0 else 0

        # --- Display KPIs ---
        st.subheader("Key Performance Indicators")
        cols1 = st.columns(5)
        cols1[0].metric("💰 Platform Revenue", f"€{final_revenue:,.2f}")
        cols1[1].metric("📈 Total Interest Accrued", f"€{total_interest:,.2f}")
        cols1[2].metric("🏆 Total Prizes", f"€{total_prizes:,.2f}")
        cols1[3].metric("💼 Avg. User Balance", f"€{avg_user_balance:,.2f}")
        cols1[4].metric("🎁 Prize Slots This Month", f"{prizes_this_month}")

        cols2 = st.columns(4)
        cols2[0].metric("👥 Depositing Users", f"{depositing_users}")
        cols2[1].metric("🎯 Goal Engagement", f"{engagement_rate:.1f}%")
        cols2[2].metric("✅ Goals Achieved", f"{achieved}")
        cols2[3].metric("❌ Goals Failed", f"{failed}")

        # --- Advanced Analytics Tab ---
        tab1, tab2 = st.tabs(["🔍 Data Explorer", "🔬 Advanced Analytics"])

        with tab1:
            st.markdown("Select a table from the dropdown to view its data and download it as a CSV.")
            display_dfs = {name: df for name, df in all_dfs.items() if not df.empty}
            if display_dfs:
                selected_table = st.selectbox("Select a table to view", list(display_dfs.keys()))
                st.dataframe(display_dfs[selected_table])
                st.download_button(label=f"Download {selected_table} as CSV", data=convert_df_to_csv(display_dfs[selected_table]), file_name=f'{selected_table.lower()}.csv', mime='text/csv')
            else:
                st.warning("No data generated.")
                
        with tab2:
            st.subheader("Detailed Annual Report")
            if not transactions_df.empty:
                # Prepare dataframes for reporting
                report_dfs = {name: df.copy() for name, df in all_dfs.items()}
                for name, df in report_dfs.items():
                    if 'date_created' in df.columns:
                        df['date_created'] = pd.to_datetime(df['date_created'])
                        df['year'] = df['date_created'].dt.year
                    elif 'date' in df.columns: # for interest log
                        df['date'] = pd.to_datetime(df['date'])
                        df['year'] = df['date'].dt.year

                # Calculate metrics per year
                years = sorted(report_dfs['Users']['year'].unique())
                report_data = []
                for year in years:
                    # Filter data for the current year
                    yearly_users = report_dfs['Users'][report_dfs['Users']['year'] <= year]
                    yearly_transactions = report_dfs['Transactions'][report_dfs['Transactions']['year'] == year]
                    yearly_goals = report_dfs['User_Goals'][report_dfs['User_Goals']['year'] == year]
                    yearly_qual = report_dfs['Qualification_Tracker'][report_dfs['Qualification_Tracker']['date_calculated'].dt.year == year]
                    yearly_interest = report_dfs['Daily_Interest_Log'][report_dfs['Daily_Interest_Log']['year'] == year]
                    
                    # Calculations
                    avg_monthly_deposits = 0
                    if not yearly_transactions.empty:
                        monthly_deposits = yearly_transactions[yearly_transactions['type'] == 'DEPOSIT'].groupby(['user_id', pd.Grouper(key='date_created', freq='M')])['amount'].sum().mean()
                        avg_monthly_deposits = monthly_deposits if not pd.isna(monthly_deposits) else 0

                    report_data.append({
                        "Year": year,
                        "Active Users at Year-End": yearly_users[yearly_users['status'] == 'ACTIVE']['user_id'].nunique(),
                        "Avg. Monthly Deposit per User": f"€{avg_monthly_deposits:,.2f}",
                        "Interest from Low-Risk": f"€{yearly_interest['low_risk_return'].sum():,.2f}",
                        "Interest from High-Risk": f"€{yearly_interest['high_risk_return'].sum():,.2f}",
                        "Total Withdrawals": f"€{yearly_transactions[yearly_transactions['type'] == 'WITHDRAWAL']['amount'].sum():,.2f}",
                        "Goal Achievers": yearly_goals[yearly_goals['status'] == 'ACHIEVED']['user_id'].nunique(),
                        "Average Goal Amount": f"€{yearly_goals['target_amount'].mean():,.2f}",
                        "Median Goal Amount": f"€{yearly_goals['target_amount'].median():,.2f}",
                        "Total Prize Money Given": f"€{yearly_transactions[yearly_transactions['type'].isin(['PRIZE_WIN', 'GOAL_REWARD'])]['amount'].sum():,.2f}",
                        "Number of Prize Winners": yearly_qual['user_id'].nunique(),
                        "Average Prize Amount": f"€{yearly_qual['prize_amount_won'].mean():,.2f}" if not yearly_qual.empty else "€0.00"
                    })
                
                st.dataframe(pd.DataFrame(report_data).set_index("Year"))

            st.subheader("Prize Pool Health")
            if not ledger_df.empty and "PRIZE_POOL" in ledger_df['account_id'].values:
                prize_pool_history = ledger_df[ledger_df['account_id'] == 'PRIZE_POOL'].copy()
                prize_pool_history['date_created'] = pd.to_datetime(prize_pool_history['date_created'])
                prize_pool_history = prize_pool_history.sort_values('date_created')
                prize_pool_history['balance'] = prize_pool_history['amount'].cumsum()
                st.line_chart(prize_pool_history, x='date_created', y='balance')
else:
    st.info("Define your investment strategy and simulation parameters, then click 'Run Simulation' to start.")
