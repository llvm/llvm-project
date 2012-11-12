; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -disable-cgp-branch-opts | FileCheck %s

%struct.pix_pos = type { i32, i32, i32, i32, i32, i32 }

@getNeighbour = external global void (i32, i32, i32, i32, %struct.pix_pos*)*, align 4 ; <void (i32, i32, i32, i32, %struct.pix_pos*)**> [#uses=2]

define void @t() nounwind {
; CHECK: t:
; CHECK:      it eq
; CHECK-NEXT: cmpeq
entry:
  %pix_a.i294 = alloca [4 x %struct.pix_pos], align 4 ; <[4 x %struct.pix_pos]*> [#uses=2]
  br i1 undef, label %land.rhs, label %lor.end

land.rhs:                                         ; preds = %entry
  br label %lor.end

lor.end:                                          ; preds = %land.rhs, %entry
  switch i32 0, label %if.end371 [
    i32 10, label %if.then366
    i32 14, label %if.then366
  ]

if.then366:                                       ; preds = %lor.end, %lor.end
  unreachable

if.end371:                                        ; preds = %lor.end
  %arrayidx56.2.i = getelementptr [4 x %struct.pix_pos]* %pix_a.i294, i32 0, i32 2 ; <%struct.pix_pos*> [#uses=1]
  %arrayidx56.3.i = getelementptr [4 x %struct.pix_pos]* %pix_a.i294, i32 0, i32 3 ; <%struct.pix_pos*> [#uses=1]
  br i1 undef, label %for.body1857, label %for.end4557

for.body1857:                                     ; preds = %if.end371
  br i1 undef, label %if.then1867, label %for.cond1933

if.then1867:                                      ; preds = %for.body1857
  unreachable

for.cond1933:                                     ; preds = %for.body1857
  br i1 undef, label %for.body1940, label %if.then4493

for.body1940:                                     ; preds = %for.cond1933
  %shl = shl i32 undef, 2                         ; <i32> [#uses=1]
  %shl1959 = shl i32 undef, 2                     ; <i32> [#uses=4]
  br i1 undef, label %if.then1992, label %if.else2003

if.then1992:                                      ; preds = %for.body1940
  %tmp14.i302 = load i32* undef                   ; <i32> [#uses=4]
  %add.i307452 = or i32 %shl1959, 1               ; <i32> [#uses=1]
  %sub.i308 = add i32 %shl, -1                    ; <i32> [#uses=4]
  call  void undef(i32 %tmp14.i302, i32 %sub.i308, i32 %shl1959, i32 0, %struct.pix_pos* undef) nounwind
  %tmp49.i309 = load void (i32, i32, i32, i32, %struct.pix_pos*)** @getNeighbour ; <void (i32, i32, i32, i32, %struct.pix_pos*)*> [#uses=1]
  call  void %tmp49.i309(i32 %tmp14.i302, i32 %sub.i308, i32 %add.i307452, i32 0, %struct.pix_pos* null) nounwind
  %tmp49.1.i = load void (i32, i32, i32, i32, %struct.pix_pos*)** @getNeighbour ; <void (i32, i32, i32, i32, %struct.pix_pos*)*> [#uses=1]
  call  void %tmp49.1.i(i32 %tmp14.i302, i32 %sub.i308, i32 undef, i32 0, %struct.pix_pos* %arrayidx56.2.i) nounwind
  call  void undef(i32 %tmp14.i302, i32 %sub.i308, i32 undef, i32 0, %struct.pix_pos* %arrayidx56.3.i) nounwind
  unreachable

if.else2003:                                      ; preds = %for.body1940
  switch i32 undef, label %if.then2015 [
    i32 10, label %if.then4382
    i32 14, label %if.then4382
  ]

if.then2015:                                      ; preds = %if.else2003
  br i1 undef, label %if.else2298, label %if.then2019

if.then2019:                                      ; preds = %if.then2015
  br i1 undef, label %if.then2065, label %if.else2081

if.then2065:                                      ; preds = %if.then2019
  br label %if.end2128

if.else2081:                                      ; preds = %if.then2019
  br label %if.end2128

if.end2128:                                       ; preds = %if.else2081, %if.then2065
  unreachable

if.else2298:                                      ; preds = %if.then2015
  br i1 undef, label %land.lhs.true2813, label %cond.end2841

land.lhs.true2813:                                ; preds = %if.else2298
  br i1 undef, label %cond.end2841, label %cond.true2824

cond.true2824:                                    ; preds = %land.lhs.true2813
  br label %cond.end2841

cond.end2841:                                     ; preds = %cond.true2824, %land.lhs.true2813, %if.else2298
  br i1 undef, label %for.cond2882.preheader, label %for.cond2940.preheader

for.cond2882.preheader:                           ; preds = %cond.end2841
  %mul3693 = shl i32 undef, 1                     ; <i32> [#uses=2]
  br i1 undef, label %if.then3689, label %if.else3728

for.cond2940.preheader:                           ; preds = %cond.end2841
  br label %for.inc3040

for.inc3040:                                      ; preds = %for.inc3040, %for.cond2940.preheader
  br label %for.inc3040

if.then3689:                                      ; preds = %for.cond2882.preheader
  %add3695 = add nsw i32 %mul3693, %shl1959       ; <i32> [#uses=1]
  %mul3697 = shl i32 %add3695, 2                  ; <i32> [#uses=2]
  %arrayidx3705 = getelementptr inbounds i16* undef, i32 1 ; <i16*> [#uses=1]
  %tmp3706 = load i16* %arrayidx3705              ; <i16> [#uses=1]
  %conv3707 = sext i16 %tmp3706 to i32            ; <i32> [#uses=1]
  %add3708 = add nsw i32 %conv3707, %mul3697      ; <i32> [#uses=1]
  %arrayidx3724 = getelementptr inbounds i16* null, i32 1 ; <i16*> [#uses=1]
  %tmp3725 = load i16* %arrayidx3724              ; <i16> [#uses=1]
  %conv3726 = sext i16 %tmp3725 to i32            ; <i32> [#uses=1]
  %add3727 = add nsw i32 %conv3726, %mul3697      ; <i32> [#uses=1]
  br label %if.end3770

if.else3728:                                      ; preds = %for.cond2882.preheader
  %mul3733 = add i32 %shl1959, 1073741816         ; <i32> [#uses=1]
  %add3735 = add nsw i32 %mul3733, %mul3693       ; <i32> [#uses=1]
  %mul3737 = shl i32 %add3735, 2                  ; <i32> [#uses=2]
  %tmp3746 = load i16* undef                      ; <i16> [#uses=1]
  %conv3747 = sext i16 %tmp3746 to i32            ; <i32> [#uses=1]
  %add3748 = add nsw i32 %conv3747, %mul3737      ; <i32> [#uses=1]
  %arrayidx3765 = getelementptr inbounds i16* null, i32 1 ; <i16*> [#uses=1]
  %tmp3766 = load i16* %arrayidx3765              ; <i16> [#uses=1]
  %conv3767 = sext i16 %tmp3766 to i32            ; <i32> [#uses=1]
  %add3768 = add nsw i32 %conv3767, %mul3737      ; <i32> [#uses=1]
  br label %if.end3770

if.end3770:                                       ; preds = %if.else3728, %if.then3689
  %vec2_y.1 = phi i32 [ %add3727, %if.then3689 ], [ %add3768, %if.else3728 ] ; <i32> [#uses=0]
  %vec1_y.2 = phi i32 [ %add3708, %if.then3689 ], [ %add3748, %if.else3728 ] ; <i32> [#uses=0]
  unreachable

if.then4382:                                      ; preds = %if.else2003, %if.else2003
  switch i32 undef, label %if.then4394 [
    i32 10, label %if.else4400
    i32 14, label %if.else4400
  ]

if.then4394:                                      ; preds = %if.then4382
  unreachable

if.else4400:                                      ; preds = %if.then4382, %if.then4382
  br label %for.cond4451.preheader

for.cond4451.preheader:                           ; preds = %for.cond4451.preheader, %if.else4400
  br label %for.cond4451.preheader

if.then4493:                                      ; preds = %for.cond1933
  unreachable

for.end4557:                                      ; preds = %if.end371
  ret void
}
