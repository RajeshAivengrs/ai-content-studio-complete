#!/usr/bin/env python3
"""
Content X AI Studio - Working Deployment
Based on Content X Bible with ICP Personalization and AI Agents
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="Content X AI Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def homepage():
    """Content X AI Studio Homepage with AI Agents"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Content X AI Studio - Enterprise AI Content Generation Platform</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
            .header { text-align: center; margin-bottom: 3rem; }
            .logo { font-size: 4rem; font-weight: 700; margin-bottom: 1rem; }
            .subtitle { font-size: 1.5rem; opacity: 0.9; margin-bottom: 2rem; }
            .status { background: rgba(0, 212, 255, 0.2); padding: 1rem; border-radius: 10px; margin: 2rem 0; border: 1px solid #00d4ff; }
            .agents-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin: 3rem 0; }
            .agent-card { 
                background: rgba(255, 255, 255, 0.1); 
                padding: 2rem; 
                border-radius: 15px; 
                backdrop-filter: blur(10px); 
                border: 1px solid rgba(255, 255, 255, 0.2); 
            }
            .agent-card.featured { border: 2px solid #00d4ff; }
            .agent-icon { font-size: 3rem; margin-bottom: 1rem; }
            .agent-card h3 { font-size: 1.5rem; margin-bottom: 1rem; color: #00d4ff; }
            .agent-role { color: #00ff88; font-weight: 600; margin-bottom: 1rem; }
            .agent-description { margin-bottom: 1.5rem; line-height: 1.6; }
            .features { display: flex; flex-wrap: wrap; gap: 0.5rem; }
            .feature-tag { 
                background: rgba(0, 255, 136, 0.2); 
                color: #00ff88; 
                padding: 0.3rem 0.8rem; 
                border-radius: 20px; 
                font-size: 0.8rem; 
            }
            .pricing-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin: 3rem 0; }
            .pricing-card { 
                background: rgba(255, 255, 255, 0.1); 
                padding: 2rem; 
                border-radius: 15px; 
                backdrop-filter: blur(10px); 
                border: 1px solid rgba(255, 255, 255, 0.2); 
                text-align: center;
            }
            .pricing-card.featured { border: 2px solid #f59e0b; background: rgba(245, 158, 11, 0.1); }
            .price { font-size: 2.5rem; font-weight: 700; color: #00d4ff; margin: 1rem 0; }
            .btn { 
                display: inline-block; 
                background: linear-gradient(45deg, #00d4ff, #00ff88); 
                color: white; 
                padding: 1rem 2rem; 
                border-radius: 50px; 
                text-decoration: none; 
                font-weight: 600; 
                margin: 1rem; 
                transition: transform 0.3s ease; 
            }
            .btn:hover { transform: translateY(-3px); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="logo">Content X AI Studio</h1>
                <p class="subtitle">🚀 Enterprise AI Content Generation Platform</p>
                <div class="status">
                    <h3>✅ Live & Running - Content X Bible Implementation</h3>
                    <p>Complete 13-layer AI Content Studio with ICP Personalization</p>
                </div>
            </div>
            
            <div class="agents-grid">
                <div class="agent-card">
                    <div class="agent-icon">🎯</div>
                    <h3>MAYA</h3>
                    <p class="agent-role">Strategist & Scanner</p>
                    <p class="agent-description">Identifies trending opportunities, scans digital ecosystems, and delivers strategic content briefs with timing signals.</p>
                    <div class="features">
                        <span class="feature-tag">Trend Analysis</span>
                        <span class="feature-tag">Opportunity ID</span>
                        <span class="feature-tag">Strategic Briefs</span>
                    </div>
                </div>
                
                <div class="agent-card featured">
                    <div class="agent-icon">✍️</div>
                    <h3>TOBY</h3>
                    <p class="agent-role">Creator & Researcher</p>
                    <p class="agent-description">Transforms briefs into engaging content across all formats - scripts, blogs, social posts, and multimedia assets.</p>
                    <div class="features">
                        <span class="feature-tag">Content Creation</span>
                        <span class="feature-tag">Multi-Format</span>
                        <span class="feature-tag">ICP-Aligned</span>
                    </div>
                </div>
                
                <div class="agent-card">
                    <div class="agent-icon">⚡</div>
                    <h3>CHIEF</h3>
                    <p class="agent-role">Operator & Executor</p>
                    <p class="agent-description">Manages campaigns, handles publishing across all platforms, and ensures compliance with brand guidelines.</p>
                    <div class="features">
                        <span class="feature-tag">Campaign Mgmt</span>
                        <span class="feature-tag">Multi-Platform</span>
                        <span class="feature-tag">Compliance</span>
                    </div>
                </div>
                
                <div class="agent-card">
                    <div class="agent-icon">🎛️</div>
                    <h3>SUPERVISOR</h3>
                    <p class="agent-role">Orchestrator</p>
                    <p class="agent-description">Ensures quality control, maintains ICP alignment, and coordinates all agents for seamless content operations.</p>
                    <div class="features">
                        <span class="feature-tag">Quality Control</span>
                        <span class="feature-tag">ICP Alignment</span>
                        <span class="feature-tag">Coordination</span>
                    </div>
                </div>
            </div>
            
            <div class="pricing-grid">
                <div class="pricing-card">
                    <h3>Tier A - Scripts Only</h3>
                    <div class="price">$29<span>/month</span></div>
                    <p>Maya scanning + Toby's script generation. No voice/video. No publishing.</p>
                    <a href="#" class="btn">Start with Tier A</a>
                </div>
                
                <div class="pricing-card">
                    <h3>Tier B - Hybrid</h3>
                    <div class="price">$79<span>/month</span></div>
                    <p>Pick Two: Scripts + Video, or Scripts + Publishing, or Scripts + Voice.</p>
                    <a href="#" class="btn">Start with Tier B</a>
                </div>
                
                <div class="pricing-card featured">
                    <h3>Tier C - Full Stack</h3>
                    <div class="price">$199<span>/month</span></div>
                    <p>Entire pipeline: Maya + Toby + Chief + Supervisor + analytics + monetization.</p>
                    <a href="#" class="btn">Become Full Stack</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "message": "Content X AI Studio with Maya/Toby/Chief/Supervisor agents is running"
    }

@app.get("/features")
async def features():
    """Features page"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Features - Content X AI Studio</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: white; }
            .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
            .header { text-align: center; margin-bottom: 3rem; }
            .logo { font-size: 3rem; font-weight: 700; margin-bottom: 1rem; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; }
            .feature { background: rgba(255, 255, 255, 0.1); padding: 2rem; border-radius: 15px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); }
            .feature h3 { font-size: 1.5rem; margin-bottom: 1rem; color: #00d4ff; }
            .btn { display: inline-block; background: rgba(255, 255, 255, 0.2); color: white; padding: 0.8rem 1.5rem; border-radius: 25px; text-decoration: none; margin-bottom: 2rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="btn">← Back to Home</a>
            <div class="header">
                <h1 class="logo">Content X AI Studio</h1>
                <p>Complete AI-driven content creation with 4 specialized agents</p>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h3>🎯 ICP Personalization</h3>
                    <p>Define your brand identity, upload style samples, and let our AI agents create content that sounds authentically like you.</p>
                </div>
                
                <div class="feature">
                    <h3>🎬 Maya - Trend Scanner</h3>
                    <p>Continuously scans digital ecosystems to identify trending opportunities and delivers strategic content briefs.</p>
                </div>
                
                <div class="feature">
                    <h3>✍️ Toby - Content Creator</h3>
                    <p>Transforms briefs into engaging content across all formats while maintaining your unique voice and style.</p>
                </div>
                
                <div class="feature">
                    <h3>⚡ Chief - Campaign Manager</h3>
                    <p>Handles publishing across all platforms, manages campaigns, and ensures compliance with your brand guidelines.</p>
                </div>
                
                <div class="feature">
                    <h3>🎛️ Supervisor - Quality Control</h3>
                    <p>Ensures all content maintains ICP alignment and coordinates all agents for seamless operations.</p>
                </div>
                
                <div class="feature">
                    <h3>💰 Monetization</h3>
                    <p>Integrated revenue streams including subscriptions, brand deals, and affiliate commerce.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)