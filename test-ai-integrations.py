#!/usr/bin/env python3
"""
Test AI Integrations for AI Content Studio
Tests OpenAI, ElevenLabs, and Google OAuth functionality
"""

import os
import asyncio
import openai
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "your-elevenlabs-api-key-here")
GOOGLE_CLIENT_ID = "81205443513-5b7lkvr1ittd87phm05kstsnjl20a7jq.apps.googleusercontent.com"

def test_openai():
    """Test OpenAI API connection"""
    print("🤖 Testing OpenAI API...")
    try:
        # Test with requests instead of OpenAI client
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Say 'Hello from AI Content Studio!' and nothing else."}
            ],
            "max_tokens": 20
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ OpenAI Response: {content}")
            return True
        else:
            print(f"❌ OpenAI Error: {response.status_code} - {response.text}")
            return False
        
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return False

def test_elevenlabs():
    """Test ElevenLabs API connection"""
    print("\n🎙️ Testing ElevenLabs API...")
    try:
        # Test getting voices
        headers = {
            "Accept": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
        
        if response.status_code == 200:
            voices = response.json()
            print(f"✅ ElevenLabs: Found {len(voices.get('voices', []))} available voices")
            return True
        else:
            print(f"❌ ElevenLabs Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ElevenLabs Error: {e}")
        return False

def test_google_oauth():
    """Test Google OAuth configuration"""
    print("\n🔐 Testing Google OAuth Configuration...")
    try:
        if GOOGLE_CLIENT_ID and len(GOOGLE_CLIENT_ID) > 50:
            print(f"✅ Google OAuth: Client ID configured ({GOOGLE_CLIENT_ID[:20]}...)")
            return True
        else:
            print("❌ Google OAuth: Client ID not configured")
            return False
    except Exception as e:
        print(f"❌ Google OAuth Error: {e}")
        return False

def test_app_endpoints():
    """Test application endpoints"""
    print("\n🌐 Testing Application Endpoints...")
    try:
        base_url = "http://localhost:8001"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint: Working")
        else:
            print(f"❌ Health endpoint: {response.status_code}")
            return False
        
        # Test homepage
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200 and "Get Started" in response.text:
            print("✅ Homepage: Working")
        else:
            print(f"❌ Homepage: {response.status_code}")
            return False
        
        # Test Google OAuth endpoint
        response = requests.get(f"{base_url}/auth/google", timeout=5)
        if response.status_code == 307:  # Redirect is expected
            print("✅ Google OAuth endpoint: Working (redirects to Google)")
        else:
            print(f"❌ Google OAuth endpoint: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Application not running on localhost:8001")
        return False
    except Exception as e:
        print(f"❌ Application Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AI Content Studio - Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Test OpenAI
    results.append(test_openai())
    
    # Test ElevenLabs
    results.append(test_elevenlabs())
    
    # Test Google OAuth
    results.append(test_google_oauth())
    
    # Test Application
    results.append(test_app_endpoints())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    tests = ["OpenAI API", "ElevenLabs API", "Google OAuth", "Application"]
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems are working perfectly!")
        print("🚀 Ready for production deployment!")
    else:
        print("⚠️  Some issues detected. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)