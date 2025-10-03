# 🚀 Content X Bible - Render Deployment Guide

## 📋 Quick Deployment Steps

### 1. Connect to Render

1. Go to [render.com](https://render.com)
2. Sign in to your account
3. Click "New +" → "Web Service"
4. Connect your GitHub account
5. Select repository: `RajeshAivengrs/content-x-bible-final`

### 2. Configure Service Settings

**Basic Settings:**
- **Name**: `content-x-bible`
- **Environment**: `Python 3`
- **Plan**: `Starter` (free tier)
- **Branch**: `main`
- **Root Directory**: `.` (leave empty)

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python ai-content-studio-production.py`

### 3. Environment Variables

Add these environment variables in Render dashboard:

**Required:**
```
OPENAI_API_KEY=your-openai-api-key-here
```

**Optional:**
```
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
STRIPE_SECRET_KEY=your-stripe-secret-key-here
```

**Auto-configured:**
```
JWT_SECRET=content-x-bible-jwt-secret-production
DATABASE_URL=sqlite:///./content_x_bible.db
PORT=8001
HOST=0.0.0.0
DEBUG=false
LOG_LEVEL=INFO
```

### 4. Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Start your application
   - Provide a public URL

### 5. Monitor Deployment

- **Build Logs**: Check build progress
- **Runtime Logs**: Monitor application health
- **Health Check**: Automatic health monitoring at `/health`

## 🔧 Manual Deployment Commands

If you prefer command-line deployment:

```bash
# Install Render CLI
npm install -g @render/cli

# Login to Render
render auth login

# Deploy using render.yaml
render deploy --service content-x-bible
```

## 📊 Expected Deployment Time

- **Build Time**: 2-3 minutes
- **Startup Time**: 30-60 seconds
- **Total Time**: 3-4 minutes

## 🌐 Access Your Application

After successful deployment:
- **URL**: `https://content-x-bible.onrender.com` (or your custom domain)
- **Health Check**: `https://your-app.onrender.com/health`
- **API Docs**: `https://your-app.onrender.com/docs`

## 🔍 Troubleshooting

### Common Issues:

1. **Build Fails**: Check `requirements.txt` for missing dependencies
2. **Startup Fails**: Verify start command and environment variables
3. **Health Check Fails**: Ensure `/health` endpoint is working
4. **Database Issues**: SQLite database will be created automatically

### Debug Commands:

```bash
# Check build logs
render logs --service content-x-bible --type build

# Check runtime logs
render logs --service content-x-bible --type deploy

# Restart service
render restart --service content-x-bible
```

## 📈 Scaling Options

### Free Tier Limits:
- **Builds**: 750 hours/month
- **Runtime**: 750 hours/month
- **Bandwidth**: 100GB/month

### Paid Plans:
- **Starter**: $7/month
- **Standard**: $25/month
- **Pro**: $85/month

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to repository
2. **Environment Variables**: Use Render's secure environment variable system
3. **HTTPS**: Automatically enabled by Render
4. **Database**: SQLite for development, PostgreSQL for production

## 📱 Features Available After Deployment

✅ **Complete AI Content Generation Platform**
✅ **All 13 Layers Implemented**
✅ **4 AI Agents (Maya, Toby, Chief, Supervisor)**
✅ **20+ API Endpoints**
✅ **Real-time WebSocket Support**
✅ **Mobile PWA Support**
✅ **Production-ready Infrastructure**

## 🎯 Post-Deployment Checklist

- [ ] Health check passes
- [ ] API endpoints respond correctly
- [ ] OpenAI integration works
- [ ] Database connections established
- [ ] Static files served correctly
- [ ] Environment variables configured
- [ ] SSL certificate active
- [ ] Performance monitoring enabled

## 📞 Support

For deployment issues:
1. Check Render dashboard logs
2. Verify environment variables
3. Test locally first
4. Contact Render support if needed

---

**Content X Bible - Complete AI Content Generation Platform**  
*Deployed on Render with full functionality*
