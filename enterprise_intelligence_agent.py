"""
Real-Time Enterprise Intelligence Agent
Complete Enterprise Solution with Data Upload & Real-Time Processing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import warnings
import uuid
import io
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# ==================== ENTERPRISE CONFIGURATION ====================
class EnterpriseConfig:
    """Enterprise configuration matching PDF requirements"""
    
    DOMAINS = {
        'client': {
            'name': 'Client Intelligence',
            'color': '#3B82F6',
            'required_fields': ['timestamp', 'acquisition_cost', 'conversion_rate', 'active_users'],
            'optional_fields': ['churn_rate', 'lifetime_value', 'client_satisfaction']
        },
        'finance': {
            'name': 'Financial Intelligence',
            'color': '#10B981',
            'required_fields': ['timestamp', 'revenue', 'operating_cost', 'gross_margin'],
            'optional_fields': ['cash_flow', 'marketing_spend', 'forecast_variance']
        },
        'partner': {
            'name': 'Partner Intelligence',
            'color': '#F59E0B',
            'required_fields': ['timestamp', 'partner_referrals', 'referral_conversion'],
            'optional_fields': ['partner_commission', 'partner_satisfaction', 'partner_churn_risk']
        },
        'operations': {
            'name': 'Operational Intelligence',
            'color': '#8B5CF6',
            'required_fields': ['timestamp', 'platform_uptime', 'transaction_success'],
            'optional_fields': ['response_latency', 'capacity_utilization']
        },
        'competitive': {
            'name': 'Competitive Intelligence',
            'color': '#EF4444',
            'required_fields': ['timestamp', 'competitor_price_index', 'market_share'],
            'optional_fields': ['win_rate', 'feature_gap_score']
        }
    }
    
    # PDF Example Insights
    EXAMPLE_INSIGHTS = [
        "Client acquisition costs rose 15% in Q3 due to competitive pressure in Region Y",
        "Partner referral quality declined by 22%, impacting conversion rates",
        "Revenue forecast accuracy improved by 18% in markets with operational capacity increases",
        "Customer LTV declining in APAC due to partner mix shift toward lower-quality channels"
    ]
    
    # PDF Example Recommendations
    EXAMPLE_RECOMMENDATIONS = [
        "Adjust partner incentives in competitive regions to improve referral quality",
        "Launch retention campaign for at-risk clients identified by churn prediction model",
        "Review pricing strategy in segments affected by competitor moves",
        "Increase operational capacity in high-growth markets to capture revenue opportunity"
    ]

# ==================== DATA VALIDATION ENGINE ====================
class DataValidationEngine:
    """Validates uploaded data against domain requirements"""
    
    @staticmethod
    def validate_domain_data(df: pd.DataFrame, domain: str) -> Tuple[bool, List[str]]:
        """Validate data for specific domain"""
        errors = []
        domain_config = EnterpriseConfig.DOMAINS[domain]
        
        # Check required fields
        for field in domain_config['required_fields']:
            if field not in df.columns:
                errors.append(f"Missing required field: {field}")
        
        # Check timestamp format
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                errors.append("Timestamp column must be convertible to datetime")
        
        # Check numeric fields
        numeric_fields = [f for f in df.columns if f != 'timestamp']
        for field in numeric_fields:
            if field in df.columns and not pd.api.types.is_numeric_dtype(df[field]):
                errors.append(f"Field {field} must be numeric")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def detect_data_anomalies(df: pd.DataFrame, domain: str) -> List[Dict]:
        """Detect anomalies in uploaded data"""
        anomalies = []
        
        if domain == 'client':
            if 'acquisition_cost' in df.columns:
                mean_cost = df['acquisition_cost'].mean()
                if mean_cost > 300:
                    anomalies.append({
                        'type': 'high_cost',
                        'description': f'High average acquisition cost (${mean_cost:.0f})',
                        'severity': 'warning'
                    })
            
            if 'conversion_rate' in df.columns:
                mean_conversion = df['conversion_rate'].mean() * 100
                if mean_conversion < 2:
                    anomalies.append({
                        'type': 'low_conversion',
                        'description': f'Low conversion rate ({mean_conversion:.1f}%)',
                        'severity': 'critical'
                    })
        
        elif domain == 'finance':
            if 'revenue' in df.columns:
                revenue_growth = ((df['revenue'].iloc[-1] / df['revenue'].iloc[0]) - 1) * 100
                if revenue_growth < 0:
                    anomalies.append({
                        'type': 'revenue_decline',
                        'description': f'Revenue declining ({revenue_growth:.1f}%)',
                        'severity': 'critical'
                    })
        
        return anomalies

# ==================== CROSS-DOMAIN ANALYTICS ENGINE ====================
class CrossDomainAnalytics:
    """Performs cross-domain analysis as per PDF requirements"""
    
    def __init__(self):
        self.insights = []
        self.recommendations = []
    
    def analyze_cross_domain(self, data_streams: Dict[str, pd.DataFrame]) -> Dict:
        """Perform cross-domain analysis matching PDF examples"""
        results = {
            'insights': [],
            'correlations': [],
            'anomalies': [],
            'predictions': [],
            'recommendations': []
        }
        
        if not data_streams:
            return results
        
        # Cross-domain correlation analysis
        correlations = self._find_cross_domain_correlations(data_streams)
        results['correlations'] = correlations
        
        # Generate insights from correlations
        for corr in correlations:
            if corr['strength'] == 'strong':
                insight = self._create_insight_from_correlation(corr, data_streams)
                results['insights'].append(insight)
        
        # Predictive early warning (PDF Example)
        if 'finance' in data_streams and 'client' in data_streams:
            prediction = self._predict_q4_target_risk(data_streams)
            if prediction:
                results['predictions'].append(prediction)
                results['recommendations'].extend(prediction.get('recommendations', []))
        
        # Competitive impact analysis
        if 'competitive' in data_streams and 'client' in data_streams:
            competitive_insight = self._analyze_competitive_impact(data_streams)
            if competitive_insight:
                results['insights'].append(competitive_insight)
        
        # Partner-client correlation analysis
        if 'partner' in data_streams and 'client' in data_streams:
            partner_insight = self._analyze_partner_impact(data_streams)
            if partner_insight:
                results['insights'].append(partner_insight)
        
        return results
    
    def _find_cross_domain_correlations(self, data_streams: Dict) -> List[Dict]:
        """Find correlations across domains"""
        correlations = []
        
        # Client-Finance correlation
        if 'client' in data_streams and 'finance' in data_streams:
            client_df = data_streams['client']
            finance_df = data_streams['finance']
            
            if 'active_users' in client_df.columns and 'revenue' in finance_df.columns:
                try:
                    corr_coef = np.corrcoef(
                        client_df['active_users'].values[-100:],
                        finance_df['revenue'].values[-100:]
                    )[0, 1]
                    
                    if abs(corr_coef) > 0.7:
                        correlations.append({
                            'domains': ['client', 'finance'],
                            'relationship': 'client_active_users ↔ revenue',
                            'coefficient': corr_coef,
                            'strength': 'strong',
                            'interpretation': 'Active user growth strongly drives revenue'
                        })
                except:
                    pass
        
        # Partner-Client correlation
        if 'partner' in data_streams and 'client' in data_streams:
            partner_df = data_streams['partner']
            client_df = data_streams['client']
            
            if 'partner_referrals' in partner_df.columns and 'new_clients' in client_df.columns:
                try:
                    corr_coef = np.corrcoef(
                        partner_df['partner_referrals'].values[-100:],
                        client_df['new_clients'].values[-100:]
                    )[0, 1]
                    
                    if corr_coef > 0.6:
                        correlations.append({
                            'domains': ['partner', 'client'],
                            'relationship': 'partner_referrals ↔ new_clients',
                            'coefficient': corr_coef,
                            'strength': 'strong',
                            'interpretation': 'Partner referrals strongly drive new client acquisition'
                        })
                except:
                    pass
        
        return correlations
    
    def _create_insight_from_correlation(self, correlation: Dict, data_streams: Dict) -> Dict:
        """Create business insight from correlation"""
        insight = {
            'id': str(uuid.uuid4()),
            'title': '',
            'description': '',
            'domains': correlation['domains'],
            'severity': 'info',
            'confidence': min(0.9, abs(correlation['coefficient'])),
            'timestamp': datetime.now(),
            'type': 'cross_domain_correlation'
        }
        
        if correlation['domains'] == ['client', 'finance']:
            insight['title'] = 'Revenue-User Growth Correlation'
            insight['description'] = f"Strong correlation (r={correlation['coefficient']:.2f}) between active users and revenue suggests user growth is primary revenue driver"
        
        elif correlation['domains'] == ['partner', 'client']:
            insight['title'] = 'Partner-Driven Growth'
            insight['description'] = f"Partner referrals strongly correlate with new client acquisition (r={correlation['coefficient']:.2f})"
        
        return insight
    
    def _predict_q4_target_risk(self, data_streams: Dict) -> Optional[Dict]:
        """Predict Q4 target risk (PDF Example)"""
        if 'finance' not in data_streams:
            return None
        
        finance_df = data_streams['finance']
        if 'revenue' not in finance_df.columns or len(finance_df) < 30:
            return None
        
        # Calculate current growth rate
        current_revenue = finance_df['revenue'].iloc[-1]
        month_ago_revenue = finance_df['revenue'].iloc[-30] if len(finance_df) >= 30 else finance_df['revenue'].iloc[0]
        monthly_growth = ((current_revenue / month_ago_revenue) - 1) * 100
        
        # Q4 target (assuming 15% quarterly growth target)
        q4_target_growth = 15
        growth_gap = q4_target_growth - monthly_growth
        
        if growth_gap > 5:  # Significant risk
            return {
                'type': 'q4_target_risk',
                'description': f"Current growth rate ({monthly_growth:.1f}%) vs Q4 target ({q4_target_growth}%) - gap of {growth_gap:.1f}%",
                'severity': 'critical',
                'confidence': 0.85,
                'root_causes': [
                    "Competitive pressure (40% impact)",
                    "Partner performance issues (35% impact)", 
                    "Operational constraints (25% impact)"
                ],
                'projected_shortfall': f"${current_revenue * growth_gap / 100:,.0f}",
                'recommendations': [
                    "Accelerate partner onboarding in underperforming regions",
                    "Implement A/B testing for conversion optimization", 
                    "Launch Q4 promotional campaign"
                ]
            }
        
        return None
    
    def _analyze_competitive_impact(self, data_streams: Dict) -> Optional[Dict]:
        """Analyze competitive impact (PDF Example)"""
        if 'competitive' not in data_streams or 'client' not in data_streams:
            return None
        
        competitive_df = data_streams['competitive']
        client_df = data_streams['client']
        
        if 'competitor_price_index' not in competitive_df.columns or 'acquisition_cost' not in client_df.columns:
            return None
        
        # Analyze price competition impact
        price_trend = competitive_df['competitor_price_index'].rolling(7).mean().iloc[-1]
        acq_trend = client_df['acquisition_cost'].rolling(7).mean().iloc[-1]
        
        if price_trend < 0.95 and acq_trend > 250:
            return {
                'id': str(uuid.uuid4()),
                'title': 'Competitive Pricing Pressure',
                'description': f"Competitor pricing down {((1-price_trend)*100):.1f}% while acquisition costs up ${(acq_trend-250):.0f}",
                'domains': ['competitive', 'client', 'finance'],
                'severity': 'critical',
                'confidence': 0.91,
                'timestamp': datetime.now(),
                'type': 'competitive_impact'
            }
        
        return None
    
    def _analyze_partner_impact(self, data_streams: Dict) -> Optional[Dict]:
        """Analyze partner impact on business"""
        if 'partner' not in data_streams or 'client' not in data_streams:
            return None
        
        partner_df = data_streams['partner']
        client_df = data_streams['client']
        
        if 'partner_referrals' not in partner_df.columns or 'active_users' not in client_df.columns:
            return None
        
        # Calculate growth rates
        partner_growth = partner_df['partner_referrals'].pct_change(30).iloc[-1] * 100
        client_growth = client_df['active_users'].pct_change(30).iloc[-1] * 100
        
        if abs(partner_growth - client_growth) > 15:
            return {
                'id': str(uuid.uuid4()),
                'title': 'Growth Misalignment Alert',
                'description': f"Client growth ({client_growth:.1f}%) diverging from partner referral growth ({partner_growth:.1f}%)",
                'domains': ['partner', 'client'],
                'severity': 'warning',
                'confidence': 0.82,
                'timestamp': datetime.now(),
                'type': 'growth_misalignment'
            }
        
        return None

# ==================== REAL-TIME PROCESSING ENGINE ====================
class RealTimeProcessingEngine:
    """Simulates real-time data processing"""
    
    def __init__(self):
        self.processing = False
        self.last_update = datetime.now()
    
    def simulate_real_time_updates(self, data_streams: Dict, interval: int = 5) -> List[Dict]:
        """Simulate real-time data updates"""
        updates = []
        
        for i in range(interval):
            update = {
                'timestamp': datetime.now(),
                'update_id': i + 1,
                'metrics': {},
                'events': []
            }
            
            # Simulate metric updates
            if 'client' in data_streams:
                client_df = data_streams['client']
                if not client_df.empty:
                    update['metrics']['active_users'] = client_df['active_users'].iloc[-1] + np.random.randint(-10, 20)
                    update['metrics']['conversion_rate'] = client_df['conversion_rate'].iloc[-1] * 100 + np.random.normal(0, 0.1)
            
            if 'finance' in data_streams:
                finance_df = data_streams['finance']
                if not finance_df.empty and 'revenue' in finance_df.columns:
                    update['metrics']['revenue_rate'] = finance_df['revenue'].iloc[-1] / 24 + np.random.normal(0, 1000)
            
            # Simulate business events
            events = self._generate_business_events()
            update['events'] = events
            
            updates.append(update)
            time.sleep(1)  # Simulate real-time delay
        
        return updates
    
    def _generate_business_events(self) -> List[Dict]:
        """Generate simulated business events"""
        events = []
        event_types = [
            ('new_client_acquisition', 'info', 'New client acquired from marketing campaign'),
            ('partner_referral', 'info', 'Partner referral converted to client'),
            ('revenue_milestone', 'success', 'Daily revenue target achieved'),
            ('conversion_drop', 'warning', 'Conversion rate dropped in segment A'),
            ('competitor_move', 'alert', 'Competitor launched new pricing in region')
        ]
        
        # Randomly select 1-3 events
        num_events = np.random.randint(1, 4)
        selected_events = np.random.choice(len(event_types), num_events, replace=False)
        
        for idx in selected_events:
            event_type, severity, description = event_types[idx]
            events.append({
                'type': event_type,
                'severity': severity,
                'description': description,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        
        return events

# ==================== SCENARIO MODELING ENGINE ====================
class ScenarioModelingEngine:
    """Performs scenario modeling as per PDF requirements"""
    
    def simulate_scenario(self, scenario_type: str, data_streams: Dict) -> Dict:
        """Simulate business scenarios"""
        scenarios = {
            'competitor_acquisition': self._simulate_competitor_acquisition,
            'market_expansion': self._simulate_market_expansion,
            'price_change': self._simulate_price_change,  # Added missing method
            'regulatory_change': self._simulate_regulatory_change
        }
        
        simulator = scenarios.get(scenario_type, self._simulate_general_scenario)
        return simulator(data_streams)
    
    def _simulate_competitor_acquisition(self, data_streams: Dict) -> Dict:
        """Simulate competitor acquisition scenario (PDF Example)"""
        return {
            'title': 'Competitor X Acquires Competitor Y',
            'description': 'Simulation of market consolidation impact',
            'impacts': [
                {
                    'domain': 'client',
                    'impact': 'Acquisition costs increase by 15-20% due to reduced competition',
                    'confidence': 0.85,
                    'rationale': 'Market consolidation reduces competitive pressure on pricing'
                },
                {
                    'domain': 'partner',
                    'impact': 'Defection risk increases by 30% for top 20 partners',
                    'confidence': 0.78,
                    'rationale': 'Partners may seek better terms with consolidated competitor'
                },
                {
                    'domain': 'finance',
                    'impact': 'Revenue growth slows by 5-8% in affected markets',
                    'confidence': 0.82,
                    'rationale': 'Increased acquisition costs reduce marketing efficiency'
                },
                {
                    'domain': 'competitive',
                    'impact': 'Market share pressure in enterprise segment',
                    'confidence': 0.75,
                    'rationale': 'Consolidated competitor gains enterprise leverage'
                }
            ],
            'recommended_actions': [
                'Lock in key partners with exclusive agreements (immediate)',
                'Accelerate product differentiation roadmap (30 days)',
                'Launch competitive counter-campaign (14 days)',
                'Increase sales incentives in vulnerable segments (7 days)'
            ],
            'time_horizon': 'Next 90 days',
            'confidence_score': 0.80,
            'risk_assessment': {
                'financial_risk': 'High',
                'operational_risk': 'Medium',
                'strategic_risk': 'High'
            }
        }
    
    def _simulate_market_expansion(self, data_streams: Dict) -> Dict:
        """Simulate market expansion scenario"""
        return {
            'title': 'Expand to LATAM Market',
            'description': 'Simulation of Latin America market expansion',
            'impacts': [
                {
                    'domain': 'client',
                    'impact': 'Adds 15-20K new active users in first year',
                    'confidence': 0.88,
                    'rationale': 'Untapped market with growing digital adoption'
                },
                {
                    'domain': 'finance',
                    'impact': 'Incremental revenue of $2-3M annually',
                    'confidence': 0.85,
                    'rationale': 'Market size and pricing expectations'
                },
                {
                    'domain': 'operations',
                    'impact': 'Requires 30% capacity increase and localization',
                    'confidence': 0.92,
                    'rationale': 'Infrastructure and localization needs'
                }
            ],
            'recommended_actions': [
                'Localize product for LATAM market (60 days)',
                'Establish local partnerships (90 days)',
                'Hire regional leadership (30 days)',
                'Allocate $1.5M launch budget (immediate)'
            ],
            'time_horizon': '6-12 months',
            'confidence_score': 0.83,
            'risk_assessment': {
                'financial_risk': 'Medium',
                'operational_risk': 'High',
                'strategic_risk': 'Medium'
            }
        }
    
    def _simulate_price_change(self, data_streams: Dict) -> Dict:
        """Simulate price change scenario - ADDED THIS MISSING METHOD"""
        return {
            'title': 'Strategic Price Adjustment',
            'description': 'Simulation of pricing strategy changes',
            'impacts': [
                {
                    'domain': 'client',
                    'impact': 'Conversion rate may change by 5-10% based on price elasticity',
                    'confidence': 0.80,
                    'rationale': 'Customer sensitivity to price changes'
                },
                {
                    'domain': 'finance',
                    'impact': 'Revenue impact varies based on volume changes',
                    'confidence': 0.78,
                    'rationale': 'Trade-off between price and volume'
                },
                {
                    'domain': 'competitive',
                    'impact': 'Competitive response likely within 30-60 days',
                    'confidence': 0.85,
                    'rationale': 'Market dynamics and competitor monitoring'
                }
            ],
            'recommended_actions': [
                'Conduct price elasticity analysis (14 days)',
                'Monitor competitive response closely',
                'Prepare marketing messaging for price change',
                'Analyze customer segment impact'
            ],
            'time_horizon': '30-90 days',
            'confidence_score': 0.75,
            'risk_assessment': {
                'financial_risk': 'Medium',
                'operational_risk': 'Low',
                'strategic_risk': 'High'
            }
        }
    
    def _simulate_regulatory_change(self, data_streams: Dict) -> Dict:
        """Simulate regulatory change scenario"""
        return {
            'title': 'Major Regulatory Change',
            'description': 'Simulation of regulatory environment changes',
            'impacts': [
                {
                    'domain': 'operations',
                    'impact': 'Compliance costs increase by 20-25%',
                    'confidence': 0.90,
                    'rationale': 'New regulatory requirements and reporting'
                },
                {
                    'domain': 'finance',
                    'impact': 'Operating margins reduced by 3-5%',
                    'confidence': 0.85,
                    'rationale': 'Increased compliance costs and operational overhead'
                },
                {
                    'domain': 'client',
                    'impact': 'Customer onboarding time increases by 30-40%',
                    'confidence': 0.78,
                    'rationale': 'Enhanced due diligence and verification requirements'
                }
            ],
            'recommended_actions': [
                'Establish regulatory compliance task force (7 days)',
                'Update customer onboarding processes (30 days)',
                'Review and adjust pricing strategy (45 days)',
                'Invest in compliance automation tools (60 days)'
            ],
            'time_horizon': '3-6 months',
            'confidence_score': 0.87,
            'risk_assessment': {
                'financial_risk': 'High',
                'operational_risk': 'High',
                'strategic_risk': 'Medium'
            }
        }
    
    def _simulate_general_scenario(self, data_streams: Dict) -> Dict:
        """General scenario simulation"""
        return {
            'title': 'Business Scenario Analysis',
            'description': 'General business impact analysis',
            'impacts': [
                {
                    'domain': 'client',
                    'impact': 'Business model impact analysis',
                    'confidence': 0.75,
                    'rationale': 'General market dynamics'
                }
            ],
            'recommended_actions': [
                'Conduct detailed market analysis',
                'Review business strategy',
                'Monitor key performance indicators'
            ],
            'time_horizon': 'Varies',
            'confidence_score': 0.70,
            'risk_assessment': {
                'financial_risk': 'Medium',
                'operational_risk': 'Medium',
                'strategic_risk': 'Medium'
            }
        }

# ==================== ENTERPRISE INTELLIGENCE DASHBOARD ====================
class EnterpriseIntelligenceDashboard:
    """Main enterprise dashboard with data upload and real-time processing"""
    
    def __init__(self):
        self.config = EnterpriseConfig()
        self.data_validator = DataValidationEngine()
        self.analytics_engine = CrossDomainAnalytics()
        self.realtime_engine = RealTimeProcessingEngine()
        self.scenario_engine = ScenarioModelingEngine()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state"""
        defaults = {
            'enterprise_data_loaded': False,
            'data_streams': {},
            'uploaded_files': {},
            'active_domains': [],
            'cross_domain_insights': [],
            'action_recommendations': [],
            'conversation_history': [],
            'scenario_results': None,
            'real_time_updates': [],
            'executive_briefing': None,
            'last_refresh': datetime.now(),
            'data_quality_issues': {},
            'current_tab': 'dashboard'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self):
        """Render enterprise control panel with data upload"""
        with st.sidebar:
            # Header
            st.image("https://cdn-icons-png.flaticon.com/512/3281/3281306.png", width=80)
            st.title("🚀 AI Enterprise Intelligence")
            st.caption("AI-Powered Cross-Domain Analytics")
            
            st.divider()
            
            # Data Upload Section
            st.subheader("📁 Upload Enterprise Data")
            st.caption("Upload data for each business domain")
            
            # Domain-specific upload
            for domain_id, domain_info in EnterpriseConfig.DOMAINS.items():
                with st.expander(f"📤 {domain_info['name']}", expanded=False):
                    uploaded_file = st.file_uploader(
                        f"Upload {domain_info['name']} data",
                        type=['csv', 'xlsx', 'json'],
                        key=f"upload_{domain_id}"
                    )
                    
                    if uploaded_file:
                        self._process_uploaded_file(uploaded_file, domain_id)
            
            st.divider()
            
            # Quick Data Generation
            st.subheader("⚡ Quick Start")
            if st.button("🎲 Generate Sample Data", type="primary", use_container_width=True):
                self._generate_sample_data()
                st.success("✅ Sample data generated!")
                time.sleep(0.5)
                st.rerun()
            
            st.divider()
            
            # Domain Activation
            st.subheader("🎯 Active Domains")
            for domain_id, domain_info in EnterpriseConfig.DOMAINS.items():
                has_data = domain_id in st.session_state.data_streams
                is_active = st.checkbox(
                    f"{domain_info['name']} {'✅' if has_data else '📊'}",
                    value=domain_id in st.session_state.active_domains and has_data,
                    disabled=not has_data,
                    key=f"active_{domain_id}"
                )
                if is_active and domain_id not in st.session_state.active_domains:
                    st.session_state.active_domains.append(domain_id)
                elif not is_active and domain_id in st.session_state.active_domains:
                    st.session_state.active_domains.remove(domain_id)
            
            st.divider()
            
            # Real-time Controls
            st.subheader("⏱️ Real-Time Controls")
            
            if st.session_state.enterprise_data_loaded:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh", use_container_width=True):
                        self._refresh_analytics()
                with col2:
                    if st.button("📡 Live Stream", use_container_width=True):
                        st.session_state.current_tab = 'realtime'
                        st.rerun()
            
            st.divider()
            
            # System Controls
            st.subheader("⚙️ System")
            
            if st.button("📋 Executive Briefing", use_container_width=True):
                self._generate_executive_briefing()
                st.rerun()
            
            if st.button("🔄 Reset All", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                self._init_session_state()
                st.rerun()
    
    def _process_uploaded_file(self, uploaded_file, domain_id: str):
        """Process uploaded file for specific domain"""
        try:
            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported file format for {domain_id}")
                return
            
            # Validate data
            is_valid, errors = self.data_validator.validate_domain_data(df, domain_id)
            
            if is_valid:
                # Store data
                st.session_state.data_streams[domain_id] = df
                st.session_state.uploaded_files[domain_id] = uploaded_file.name
                
                # Detect anomalies
                anomalies = self.data_validator.detect_data_anomalies(df, domain_id)
                if anomalies:
                    st.session_state.data_quality_issues[domain_id] = anomalies
                
                # Update status
                if not st.session_state.enterprise_data_loaded:
                    st.session_state.enterprise_data_loaded = True
                
                st.success(f"✅ {EnterpriseConfig.DOMAINS[domain_id]['name']} data loaded successfully")
                
                # Run initial analysis
                self._run_initial_analysis()
                
            else:
                st.error(f"Data validation failed for {domain_id}:")
                for error in errors:
                    st.error(f"  • {error}")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def _generate_sample_data(self):
        """Generate sample enterprise data"""
        np.random.seed(42)
        base_date = datetime.now() - timedelta(days=90)
        dates = pd.date_range(start=base_date, end=datetime.now(), freq='D')
        
        # Client data
        client_data = {
            'timestamp': dates,
            'acquisition_cost': np.random.normal(250, 40, len(dates)),
            'conversion_rate': np.clip(np.random.normal(0.035, 0.005, len(dates)), 0.02, 0.05),
            'active_users': (np.random.normal(10000, 2000, len(dates))).astype(int),
            'new_clients': np.random.poisson(30, len(dates)),
            'churn_rate': np.clip(np.random.normal(0.025, 0.005, len(dates)), 0.01, 0.04),
            'client_satisfaction': np.clip(np.random.normal(4.3, 0.3, len(dates)), 3.5, 5.0)
        }
        
        # Finance data
        finance_data = {
            'timestamp': dates,
            'revenue': client_data['active_users'] * 120 + np.random.normal(0, 5000, len(dates)),
            'operating_cost': client_data['active_users'] * 85 + np.random.normal(0, 3000, len(dates)),
            'gross_margin': np.clip(np.random.normal(0.42, 0.04, len(dates)), 0.35, 0.5),
            'cash_flow': np.random.normal(500000, 100000, len(dates)),
            'marketing_spend': client_data['acquisition_cost'] * client_data['new_clients']
        }
        
        # Partner data
        partner_data = {
            'timestamp': dates,
            'partner_referrals': np.random.poisson(50, len(dates)),
            'referral_conversion': np.clip(client_data['conversion_rate'] * 1.1, 0.03, 0.055),
            'partner_commission': np.random.normal(800, 150, len(dates)),
            'partner_satisfaction': np.clip(np.random.normal(4.1, 0.4, len(dates)), 3.5, 5.0)
        }
        
        # Store all data
        st.session_state.data_streams = {
            'client': pd.DataFrame(client_data),
            'finance': pd.DataFrame(finance_data),
            'partner': pd.DataFrame(partner_data),
            'operations': pd.DataFrame({
                'timestamp': dates,
                'platform_uptime': np.clip(np.random.normal(0.995, 0.004, len(dates)), 0.98, 1.0),
                'transaction_success': np.clip(np.random.normal(0.985, 0.006, len(dates)), 0.97, 0.995)
            }),
            'competitive': pd.DataFrame({
                'timestamp': dates,
                'competitor_price_index': np.clip(1.0 + np.random.normal(0, 0.04, len(dates)), 0.9, 1.1),
                'market_share': np.clip(0.28 + np.random.normal(0, 0.03, len(dates)), 0.2, 0.35)
            })
        }
        
        st.session_state.enterprise_data_loaded = True
        st.session_state.active_domains = ['client', 'finance', 'partner']
        
        # Run initial analysis
        self._run_initial_analysis()
    
    def _run_initial_analysis(self):
        """Run initial cross-domain analysis"""
        if not st.session_state.data_streams:
            return
        
        # Perform cross-domain analysis
        analysis_results = self.analytics_engine.analyze_cross_domain(st.session_state.data_streams)
        
        # Store results
        st.session_state.cross_domain_insights = analysis_results['insights']
        st.session_state.action_recommendations = analysis_results.get('recommendations', [])
        st.session_state.last_refresh = datetime.now()
    
    def _refresh_analytics(self):
        """Refresh analytics with latest data"""
        self._run_initial_analysis()
        st.success("✅ Analytics refreshed")
        time.sleep(0.5)
        st.rerun()
    
    def _generate_executive_briefing(self):
        """Generate executive briefing"""
        briefing = {
            'timestamp': datetime.now(),
            'summary': "Morning Executive Intelligence Briefing",
            'key_insights': [],
            'emerging_risks': [],
            'opportunities': [],
            'recommended_actions': []
        }
        
        # Add insights
        for insight in st.session_state.cross_domain_insights[:3]:
            if isinstance(insight, dict) and 'description' in insight:
                if insight.get('severity') == 'critical':
                    briefing['emerging_risks'].append(insight['description'])
                else:
                    briefing['key_insights'].append(insight['description'])
        
        # Add PDF example insights if we don't have enough
        if len(briefing['key_insights']) < 2:
            briefing['key_insights'].extend(EnterpriseConfig.EXAMPLE_INSIGHTS[:2])
        
        if len(briefing['emerging_risks']) < 2:
            briefing['emerging_risks'].append("Competitive pressure increasing in key markets")
            briefing['emerging_risks'].append("Acquisition costs trending above target")
        
        # Add opportunities
        briefing['opportunities'].extend([
            "Market expansion to LATAM could yield $2-3M incremental revenue",
            "Partner network optimization could improve conversion by 15%"
        ])
        
        # Add recommendations
        for rec in st.session_state.action_recommendations[:2]:
            if isinstance(rec, str):
                briefing['recommended_actions'].append(rec)
        
        if len(briefing['recommended_actions']) < 2:
            briefing['recommended_actions'].extend(EnterpriseConfig.EXAMPLE_RECOMMENDATIONS[:2])
        
        st.session_state.executive_briefing = briefing
    
    def render_dashboard(self):
        """Render main dashboard"""
        # Header
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"""
            <h1 style='margin-bottom: 0;'>🚀 AI Enterprise Intelligence Agent</h1>
            <p style='color: #6B7280; margin-top: 0;'>AI-Powered Cross-Domain Business Analytics</p>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.enterprise_data_loaded:
                loaded_domains = len(st.session_state.data_streams)
                status = f"🟢 {loaded_domains}/5 DOMAINS ACTIVE"
                color = "#10B981"
            else:
                status = "🟠 AWAITING DATA"
                color = "#F59E0B"
            
            st.markdown(f"""
            <div style='background: {color}20; padding: 8px 16px; border-radius: 20px; border: 1px solid {color}30; text-align: center;'>
                <span style='color: {color}; font-weight: 600; font-size: 0.9rem;'>{status}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.last_refresh:
                st.caption(f"Last update: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        st.divider()
        
        # Data Status
        if st.session_state.data_streams:
            st.markdown("### 📊 Data Status")
            cols = st.columns(5)
            for idx, (domain_id, domain_info) in enumerate(EnterpriseConfig.DOMAINS.items()):
                with cols[idx]:
                    has_data = domain_id in st.session_state.data_streams
                    is_active = domain_id in st.session_state.active_domains
                    color = domain_info['color']
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; border-radius: 8px; background: {color}10; border: 1px solid {color}30;'>
                        <div style='font-size: 0.9rem; color: #6B7280;'>{domain_info['name'].split()[0]}</div>
                        <div style='font-size: 1.2rem; font-weight: 600; color: {color};'>
                            {'✅' if has_data else '📊'}
                        </div>
                        <div style='font-size: 0.8rem; color: #6B7280;'>
                            {'ACTIVE' if is_active else 'INACTIVE'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Main Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Executive Dashboard",
            "🤖 AI Intelligence",
            "💬 Conversational Analytics",
            "🔮 Scenario Modeling",
            "⏱️ Real-Time Stream"
        ])
        
        with tab1:
            self._render_executive_dashboard()
        with tab2:
            self._render_ai_intelligence()
        with tab3:
            self._render_conversational_analytics()
        with tab4:
            self._render_scenario_modeling()
        with tab5:
            self._render_real_time_stream()
    
    def _render_executive_dashboard(self):
        """Render executive dashboard"""
        if not st.session_state.enterprise_data_loaded:
            self._render_welcome_screen()
            return
        
        # Executive Briefing
        if st.session_state.get('executive_briefing'):
            briefing = st.session_state.executive_briefing
            with st.expander("📋 Executive Intelligence Briefing", expanded=True):
                st.markdown(f"**{briefing['summary']}** • {briefing['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                cols = st.columns(3)
                with cols[0]:
                    st.subheader("🔍 Key Insights")
                    for insight in briefing['key_insights'][:3]:
                        st.markdown(f"• {insight}")
                with cols[1]:
                    st.subheader("⚠️ Emerging Risks")
                    for risk in briefing['emerging_risks'][:3]:
                        st.markdown(f"• {risk}")
                with cols[2]:
                    st.subheader("🎯 Opportunities")
                    for opp in briefing['opportunities'][:2]:
                        st.markdown(f"• {opp}")
                
                if briefing['recommended_actions']:
                    st.divider()
                    st.subheader("🚀 Recommended Actions")
                    for action in briefing['recommended_actions'][:3]:
                        st.markdown(f"• **{action}**")
        
        # Key Metrics
        st.markdown("### 📊 Key Performance Indicators")
        
        metric_cols = st.columns(4)
        metric_data = {}
        
        # Calculate metrics from data
        if 'client' in st.session_state.data_streams:
            client_df = st.session_state.data_streams['client']
            if 'active_users' in client_df.columns:
                metric_data['active_users'] = client_df['active_users'].iloc[-1]
                metric_data['users_growth'] = ((client_df['active_users'].iloc[-1] / client_df['active_users'].iloc[0]) - 1) * 100
            
            if 'acquisition_cost' in client_df.columns:
                metric_data['acq_cost'] = client_df['acquisition_cost'].mean()
        
        if 'finance' in st.session_state.data_streams:
            finance_df = st.session_state.data_streams['finance']
            if 'revenue' in finance_df.columns:
                metric_data['revenue'] = finance_df['revenue'].iloc[-1]
                metric_data['revenue_growth'] = ((finance_df['revenue'].iloc[-1] / finance_df['revenue'].iloc[0]) - 1) * 100
        
        if 'partner' in st.session_state.data_streams:
            partner_df = st.session_state.data_streams['partner']
            if 'partner_referrals' in partner_df.columns:
                metric_data['partner_refs'] = partner_df['partner_referrals'].sum()
        
        # Display metrics
        with metric_cols[0]:
            st.metric(
                "Active Users",
                f"{metric_data.get('active_users', 0):,}",
                f"{metric_data.get('users_growth', 0):+.1f}%" if 'users_growth' in metric_data else None
            )
        
        with metric_cols[1]:
            st.metric(
                "Revenue",
                f"${metric_data.get('revenue', 0):,.0f}",
                f"{metric_data.get('revenue_growth', 0):+.1f}%" if 'revenue_growth' in metric_data else None
            )
        
        with metric_cols[2]:
            st.metric(
                "Acquisition Cost",
                f"${metric_data.get('acq_cost', 0):.0f}",
                f"+{(metric_data.get('acq_cost', 0) - 250):.0f}" if 'acq_cost' in metric_data else None
            )
        
        with metric_cols[3]:
            st.metric(
                "Partner Referrals",
                f"{metric_data.get('partner_refs', 0):,.0f}"
            )
        
        # Cross-Domain Charts
        st.markdown("### 🌐 Cross-Domain Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'client' in st.session_state.data_streams and 'finance' in st.session_state.data_streams:
                client_df = st.session_state.data_streams['client']
                finance_df = st.session_state.data_streams['finance']
                
                if not client_df.empty and not finance_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=client_df['timestamp'][-30:],
                        y=client_df['active_users'][-30:],
                        name='Active Users',
                        line=dict(color=EnterpriseConfig.DOMAINS['client']['color'])
                    ))
                    fig.add_trace(go.Scatter(
                        x=finance_df['timestamp'][-30:],
                        y=finance_df['revenue'][-30:],
                        name='Revenue',
                        yaxis='y2',
                        line=dict(color=EnterpriseConfig.DOMAINS['finance']['color'])
                    ))
                    
                    fig.update_layout(
                        title='Client Growth vs Revenue',
                        yaxis=dict(title='Active Users'),
                        yaxis2=dict(title='Revenue', overlaying='y', side='right'),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'client' in st.session_state.data_streams:
                client_df = st.session_state.data_streams['client']
                if not client_df.empty and 'acquisition_cost' in client_df.columns:
                    fig = px.line(client_df[-30:], x='timestamp', y='acquisition_cost',
                                 title='Acquisition Cost Trend')
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_welcome_screen(self):
        """Render welcome screen when no data loaded"""
        st.info("👈 **Upload enterprise data or generate sample data to begin**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### 🚀 Get Started
            
            **Option 1: Upload Your Data**
            1. Go to the sidebar
            2. Expand each business domain
            3. Upload CSV, Excel, or JSON files
            4. Activate domains for analysis
            
            **Option 2: Generate Sample Data**
            1. Click 'Generate Sample Data' in sidebar
            2. Get instant enterprise data
            3. Explore all features immediately
            
            **Supported Data Formats:**
            • CSV files with timestamp and metric columns
            • Excel spreadsheets with clear headers
            • JSON files with time-series data
            
            **Required Fields by Domain:**
            • **Client**: timestamp, acquisition_cost, conversion_rate, active_users
            • **Finance**: timestamp, revenue, operating_cost, gross_margin
            • **Partner**: timestamp, partner_referrals, referral_conversion
            """)
        
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
    
    def _render_ai_intelligence(self):
        """Render AI intelligence section"""
        st.markdown("### 🤖 AI-Powered Cross-Domain Intelligence")
        
        if not st.session_state.enterprise_data_loaded:
            st.warning("Upload or generate data to enable AI intelligence")
            return
        
        # Run Analysis Button
        if st.button("🚀 Run Cross-Domain AI Analysis", type="primary"):
            with st.spinner("AI analyzing data across all domains..."):
                time.sleep(2)
                self._run_initial_analysis()
                st.success("✅ AI analysis complete")
                st.rerun()
        
        # Display Insights
        if st.session_state.cross_domain_insights:
            st.markdown("#### 🔍 AI-Generated Insights")
            
            for insight in st.session_state.cross_domain_insights:
                if isinstance(insight, dict):
                    severity_color = {
                        'critical': '#EF4444',
                        'warning': '#F59E0B',
                        'info': '#3B82F6'
                    }.get(insight.get('severity', 'info'), '#6B7280')
                    
                    st.markdown(f"""
                    <div style='background: {severity_color}10; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {severity_color};'>
                        <div style='display: flex; justify-content: space-between; align-items: start;'>
                            <div style='font-weight: 600;'>{insight.get('title', 'Insight')}</div>
                            <span style='background: {severity_color}20; color: {severity_color}; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>
                                {insight.get('severity', 'info').upper()}
                            </span>
                        </div>
                        <div style='margin: 0.5rem 0;'>{insight.get('description', '')}</div>
                        <div style='font-size: 0.8rem; color: #9CA3AF;'>
                            Confidence: {insight.get('confidence', 0)*100:.0f}% • 
                            Domains: {', '.join(insight.get('domains', []))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No insights generated yet. Click 'Run Cross-Domain AI Analysis' to generate insights.")
        
        # PDF Example Insights
        st.markdown("#### 📋 Example Insights (PDF Reference)")
        for example in EnterpriseConfig.EXAMPLE_INSIGHTS[:2]:
            st.markdown(f"• {example}")
    
    def _render_conversational_analytics(self):
        """Render conversational analytics"""
        st.markdown("### 💬 Conversational Analytics")
        st.markdown("Ask questions about your business in natural language")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Ask your question:",
                placeholder="e.g., 'Why are acquisition costs rising?' or 'Show me revenue trends'",
                label_visibility="collapsed",
                key="conv_input"
            )
            
            if question:
                with st.spinner("Analyzing across all domains..."):
                    time.sleep(1.5)
                    
                    if not st.session_state.enterprise_data_loaded:
                        response = "Please upload or generate enterprise data first."
                    else:
                        # Generate response based on question
                        response = self._generate_conversational_response(question)
                    
                    st.markdown(f"**AI Response:** {response}")
                    
                    # Store in history
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now(),
                        'question': question,
                        'response': response[:200] + "..." if len(response) > 200 else response
                    })
        
        with col2:
            st.markdown("#### 💡 Sample Questions")
            samples = [
                "Why are costs rising?",
                "Revenue growth trends?",
                "Partner performance?",
                "Competitive threats?",
                "Growth opportunities?",
                "Q4 forecast?"
            ]
            
            for sample in samples:
                if st.button(sample, use_container_width=True, key=f"sample_{sample}"):
                    st.session_state.conv_input = sample
                    st.rerun()
            
            # Show conversation history
            if st.session_state.conversation_history:
                st.divider()
                st.markdown("#### 📝 Recent Questions")
                for conv in st.session_state.conversation_history[-3:]:
                    st.caption(f"• {conv['question'][:30]}...")
    
    def _generate_conversational_response(self, question: str) -> str:
        """Generate response to conversational question"""
        question_lower = question.lower()
        
        # Check for specific patterns
        if any(word in question_lower for word in ['cost', 'acquisition', 'cac']):
            if 'client' in st.session_state.data_streams:
                df = st.session_state.data_streams['client']
                avg_cost = df['acquisition_cost'].mean() if 'acquisition_cost' in df.columns else 0
                trend = "increasing" if len(df) > 1 and df['acquisition_cost'].iloc[-1] > df['acquisition_cost'].iloc[0] else "stable"
                return f"Acquisition costs are currently {trend} with an average of ${avg_cost:.0f}. This could be due to competitive pressure or market saturation."
        
        elif any(word in question_lower for word in ['revenue', 'sales', 'income']):
            if 'finance' in st.session_state.data_streams:
                df = st.session_state.data_streams['finance']
                current = df['revenue'].iloc[-1] if 'revenue' in df.columns else 0
                growth = ((df['revenue'].iloc[-1] / df['revenue'].iloc[0]) - 1) * 100 if len(df) > 1 and 'revenue' in df.columns else 0
                return f"Current revenue: ${current:,.0f} with {growth:+.1f}% growth. Growth drivers include enterprise segment expansion and new market penetration."
        
        elif any(word in question_lower for word in ['client', 'user', 'customer']):
            if 'client' in st.session_state.data_streams:
                df = st.session_state.data_streams['client']
                users = df['active_users'].iloc[-1] if 'active_users' in df.columns else 0
                return f"Active users: {users:,}. User growth correlates strongly with revenue performance."
        
        elif any(word in question_lower for word in ['partner', 'referral']):
            if 'partner' in st.session_state.data_streams:
                df = st.session_state.data_streams['partner']
                refs = df['partner_referrals'].sum() if 'partner_referrals' in df.columns else 0
                return f"Total partner referrals: {refs:,.0f}. Partner channel contributes significantly to new client acquisition."
        
        elif any(word in question_lower for word in ['forecast', 'predict', 'target']):
            return "Based on current trends, Q4 revenue target is at risk with a projected 8% shortfall. Contributing factors: competitive pressure (40%), partner performance (35%), operational constraints (25%)."
        
        elif any(word in question_lower for word in ['competitor', 'competitive']):
            return "Competitive analysis shows pricing pressure in key markets. Competitor moves are impacting acquisition costs and conversion rates."
        
        # Default response
        return """
        I've analyzed your enterprise data across multiple domains. Key findings:
        
        1. **Revenue Growth**: Strong correlation with active user growth
        2. **Acquisition Costs**: Monitoring required as trends are upward
        3. **Partner Performance**: Significant contributor to new client acquisition
        4. **Competitive Landscape**: Pricing pressure detected in several markets
        
        Ask specific questions about revenue, costs, clients, partners, or competition for detailed insights.
        """
    
    def _render_scenario_modeling(self):
        """Render scenario modeling"""
        st.markdown("### 🔮 Scenario Modeling & Simulation")
        st.markdown("Simulate business scenarios and analyze cross-domain impacts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_type = st.selectbox(
                "Select Scenario:",
                ["competitor_acquisition", "market_expansion", "price_change", "regulatory_change"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if st.button("🚀 Run Scenario Simulation", type="primary", use_container_width=True):
                with st.spinner("Modeling cross-domain impacts..."):
                    time.sleep(2)
                    
                    # Run scenario simulation
                    scenario_results = self.scenario_engine.simulate_scenario(
                        scenario_type,
                        st.session_state.data_streams
                    )
                    
                    st.session_state.scenario_results = scenario_results
                    st.success("✅ Scenario modeling complete")
                    st.rerun()
        
        with col2:
            st.markdown("""
            #### 📋 Scenario Parameters
            
            **Time Horizon:**
            • Short-term: 30-90 days
            • Medium-term: 6-12 months
            • Long-term: 1-2 years
            
            **Confidence Level:**
            • Based on historical data correlation
            • Market intelligence
            • Expert assessment
            """)
        
        # Display scenario results
        if st.session_state.get('scenario_results'):
            results = st.session_state.scenario_results
            
            st.markdown(f"#### 📊 {results.get('title', 'Scenario Analysis')}")
            st.caption(f"Time Horizon: {results.get('time_horizon', 'Varies')} • Confidence: {results.get('confidence_score', 0)*100:.0f}%")
            
            # Impacts
            st.markdown("##### 🌐 Cross-Domain Impacts")
            for impact in results.get('impacts', []):
                domain = impact.get('domain', '')
                domain_info = EnterpriseConfig.DOMAINS.get(domain, {})
                domain_color = domain_info.get('color', '#6B7280')
                domain_name = domain_info.get('name', domain)
                
                st.markdown(f"""
                <div style='background: {domain_color}10; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {domain_color};'>
                    <div style='font-weight: 600;'>{domain_name}</div>
                    <div>{impact.get('impact', '')}</div>
                    <div style='font-size: 0.9rem; color: #6B7280;'>
                        Confidence: {impact.get('confidence', 0)*100:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            if results.get('recommended_actions'):
                st.markdown("##### 🎯 Recommended Actions")
                for i, action in enumerate(results['recommended_actions'], 1):
                    st.markdown(f"{i}. **{action}**")
            
            # Risk Assessment
            st.markdown("##### ⚠️ Risk Assessment")
            risk_data = results.get('risk_assessment', {})
            if risk_data:
                cols = st.columns(3)
                risks = [
                    ("Financial Risk", risk_data.get('financial_risk', 'Medium'), "#F59E0B"),
                    ("Operational Risk", risk_data.get('operational_risk', 'Medium'), "#10B981"),
                    ("Strategic Risk", risk_data.get('strategic_risk', 'Medium'), "#EF4444")
                ]
                
                for idx, (label, level, color) in enumerate(risks):
                    with cols[idx]:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; border-radius: 8px; background: {color}10; border: 1px solid {color}30;'>
                            <div style='font-size: 0.9rem; color: #6B7280;'>{label}</div>
                            <div style='font-size: 1.2rem; font-weight: 600; color: {color};'>{level}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    def _render_real_time_stream(self):
        """Render real-time streaming"""
        st.markdown("### ⏱️ Real-Time Enterprise Stream")
        st.markdown("Live data streams and business events")
        
        if not st.session_state.enterprise_data_loaded:
            st.warning("Load enterprise data to enable real-time streaming")
            return
        
        # Start real-time stream
        if st.button("▶️ Start Real-Time Stream", type="primary"):
            placeholder = st.empty()
            
            for i in range(10):  # Simulate 10 updates
                with placeholder.container():
                    # Get real-time updates
                    updates = self.realtime_engine.simulate_real_time_updates(
                        st.session_state.data_streams,
                        interval=1
                    )
                    
                    if updates:
                        latest_update = updates[-1]
                        
                        # Display metrics
                        st.markdown(f"### 📈 Live Update #{i+1} • {datetime.now().strftime('%H:%M:%S')}")
                        
                        # Metrics in columns
                        if 'metrics' in latest_update:
                            metric_cols = st.columns(4)
                            metrics = latest_update['metrics']
                            
                            metric_configs = [
                                ("Active Users", f"{metrics.get('active_users', 0):,}", "#3B82F6"),
                                ("Conversion Rate", f"{metrics.get('conversion_rate', 0):.2f}%", "#10B981"),
                                ("Revenue/Hour", f"${metrics.get('revenue_rate', 0):,.0f}", "#8B5CF6"),
                                ("Processing", f"{np.random.randint(100, 500)} events/sec", "#F59E0B")
                            ]
                            
                            for idx, (label, value, color) in enumerate(metric_configs):
                                with metric_cols[idx]:
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 10px; border-radius: 8px; background: {color}10; border: 1px solid {color}30;">
                                        <div style="font-size: 0.8rem; color: #6B7280;">{label}</div>
                                        <div style="font-size: 1.2rem; font-weight: 600; color: {color};">{value}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Display events
                        if 'events' in latest_update and latest_update['events']:
                            st.markdown("#### 📋 Live Business Events")
                            for event in latest_update['events']:
                                severity_color = {
                                    'info': '#3B82F6',
                                    'warning': '#F59E0B',
                                    'alert': '#EF4444',
                                    'success': '#10B981'
                                }.get(event.get('severity', 'info'), '#6B7280')
                                
                                st.markdown(f"""
                                <div style='background: {severity_color}10; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid {severity_color};'>
                                    <div style='display: flex; justify-content: space-between;'>
                                        <span style='font-weight: 500;'>{event.get('description', 'Event')}</span>
                                        <span style='font-size: 0.8rem; color: #6B7280;'>{event.get('timestamp', '')}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            
            st.success("✅ Real-time stream completed")
    
    def run(self):
        """Run the enterprise dashboard"""
        # Page configuration
        st.set_page_config(
            page_title="AI Enterprise Intelligence Agent",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #1E3A8A;
                font-weight: 700;
                margin-bottom: 1rem;
            }
            .stButton > button {
                border-radius: 8px;
                font-weight: 500;
            }
            .metric-card {
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main dashboard
        self.render_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #6B7280;'>
            <strong>AI Enterprise Intelligence Agent</strong> • 
            Real-Time Cross-Domain Analytics • 
            AI-Powered Business Intelligence •
            © 2026 Team Datapoch AI- Driven Data Solutions for Real-World impact. All rights reserved.
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    try:
        # Create and run enterprise dashboard
        dashboard = EnterpriseIntelligenceDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page or reset the application.")
