# Exercise 4 Video Recording Instructions

## What You Have

1. **slides.tex** - Beamer presentation (14 slides, ~7-8 minutes)
2. **video_script.md** - Natural-sounding script to narrate over the slides

## Quick Start

### Step 1: Compile the Slides

In Overleaf (same project as main.tex):
1. Upload `slides.tex` if not already there
2. Click "Compile"
3. The PDF will use plots from `output/` directory (already there)
4. Click the "Present" button (or download PDF for local presentation)

### Step 2: Record Your Video

**Option A: Record in Overleaf (Recommended)**
1. Enter presentation mode in Overleaf
2. Use screen recording software (OBS, Zoom, Loom)
3. Start recording
4. Advance through slides while reading script naturally
5. Stop recording when done

**Option B: Download and Present Locally**
1. Download slides.pdf from Overleaf
2. Open in presentation mode (F5 in most viewers)
3. Record with OBS Studio or similar
4. Follow script timing

### Step 3: Natural Delivery

**Key principle:** Don't read the bullet points on slides verbatim!

The slides are visual aids. The script in `video_script.md` has expanded, conversational versions:

**Slide says:** "Recursive partitioning"
**You say:** "Tree-based methods embody what I call 'divide and conquer.' They ask sequential yes-no questions to recursively partition the input space..."

See `video_script.md` for full narration with timing marks.

### Step 4: Upload and Link

1. Export video as MP4
2. Upload to:
   - **YouTube (unlisted)** - recommended for permanent hosting
   - **Google Drive** - set to "Anyone with link can view"
3. Get shareable link
4. Update `main.tex` line 868:
   ```latex
   \url{https://youtu.be/YOUR_VIDEO_ID}
   ```
5. Recompile main.tex
6. Verify link works before final submission

## Slide Overview

### Part 1: Philosophical Foundation (2 min)
- **Slide 2:** Tree-based vs kernel methods (partition vs transform)
- **Slide 3:** Bias-variance evolution (system property)
- **Slide 4:** Most elegant insight (constraints encode structure)

### Part 2: Practical Wisdom (3 min)
- **Slide 5:** Surprising finding - linear SVR dominance
- **Slide 6:** The paradox - RF saw non-linearity but lost (shows PDP plot)
- **Slide 7:** MNIST three key insights
- **Slide 8:** Misclassified examples (shows actual errors)
- **Slide 9:** Six practical lessons learned

### Part 3: The Big Picture (2 min)
- **Slide 10:** When to prefer SVMs vs trees
- **Slide 11:** Limitations encountered
- **Slide 12:** Transformation: algorithm → data centric
- **Slide 13:** Key takeaway (confidence to choose simplicity)

## Timing Guidelines

Total: ~7-8 minutes

- **Slides 1-4:** 2:00 (philosophical)
- **Slides 5-9:** 3:00 (practical wisdom)
- **Slides 10-13:** 2:00 (big picture)
- **Slide 14:** 0:15 (closing)

Each slide has specific timing in `video_script.md` (e.g., "SLIDE 5: 2:15-3:00")

## Recording Setup Checklist

- [ ] Slides compiled and working in Overleaf
- [ ] Plots visible in slides (partial_dependence.png, svm_misclassified.png)
- [ ] Recording software tested (audio levels good)
- [ ] Script printed or on second monitor
- [ ] Timer visible during recording
- [ ] Practice run completed (aim for 7:00-7:30 target)

## Tips for Natural Delivery

1. **Conversational tone** - imagine explaining to a colleague
2. **Pause after key insights** - let points land
3. **Vary your tone:**
   - Thoughtful for philosophical parts
   - Excited for surprising findings
   - Reflective for big picture
4. **Use transitions from script:**
   - "Let me share..."
   - "Here's what surprised me..."
   - "This taught me..."
5. **Point to slides** (if webcam recording) - helps engagement

## Troubleshooting

**Slides won't compile?**
- Make sure `output/` directory with plots exists in Overleaf
- Check that partial_dependence.png and svm_misclassified.png are uploaded

**Recording too long?**
- See "If Recording Goes Long" section in video_script.md
- Can cut ~30 seconds by shortening examples

**Recording too short?**
- See "If Recording Goes Short" section in video_script.md
- Can add ~30 seconds with expanded explanations

**Need to re-record a section?**
- Record each part separately (Part 1, 2, 3)
- Stitch together with video editing software
- Or just re-record the whole thing (only 7 minutes!)

## After Recording

1. Watch once for quality check
2. Verify audio is clear throughout
3. Check that slide transitions are smooth
4. Upload to hosting platform
5. Test link in private/incognito browser
6. Update main.tex with link
7. Submit!

Good luck! You've got this. 🎬
