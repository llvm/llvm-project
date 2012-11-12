; RUN: llc < %s -mtriple=i386-apple-darwin -mcpu=core2
; rdar://7857830

%0 = type opaque
%1 = type opaque

define void @t(%0* %self, i8* nocapture %_cmd, %1* %scroller, i32 %hitPart, float %multiplier) nounwind optsize ssp {
entry:
  switch i32 %hitPart, label %if.else [
    i32 7, label %if.then
    i32 8, label %if.then
  ]

if.then:                                          ; preds = %entry, %entry
  %tmp69 = load float* null, align 4              ; <float> [#uses=1]
  %cmp19 = icmp eq %1* null, %scroller            ; <i1> [#uses=2]
  %cond = select i1 %cmp19, float %tmp69, float 0.000000e+00 ; <float> [#uses=1]
  %call36 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*)*)(i8* undef, i8* undef) nounwind optsize ; <i64> [#uses=2]
  br i1 %cmp19, label %cond.true32, label %cond.false39

cond.true32:                                      ; preds = %if.then
  %sroa.store.elt68 = lshr i64 %call36, 32        ; <i64> [#uses=1]
  %0 = trunc i64 %sroa.store.elt68 to i32         ; <i32> [#uses=1]
  br label %cond.end47

cond.false39:                                     ; preds = %if.then
  %1 = trunc i64 %call36 to i32                   ; <i32> [#uses=1]
  br label %cond.end47

cond.end47:                                       ; preds = %cond.false39, %cond.true32
  %cond48.in = phi i32 [ %0, %cond.true32 ], [ %1, %cond.false39 ] ; <i32> [#uses=1]
  %cond48 = bitcast i32 %cond48.in to float       ; <float> [#uses=1]
  %div = fdiv float %cond, undef                  ; <float> [#uses=1]
  %div58 = fdiv float %div, %cond48               ; <float> [#uses=1]
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, float)*)(i8* undef, i8* undef, float %div58) nounwind optsize
  ret void

if.else:                                          ; preds = %entry
  ret void
}

declare i8* @objc_msgSend(i8*, i8*, ...)
