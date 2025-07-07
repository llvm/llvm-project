; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check for a reasonable output. This testcase used to crash.
; CHECK: .size fred

target triple = "hexagon"

define void @fred() local_unnamed_addr #0 {
entry:
  %0 = load i64, ptr undef, align 8
  %shr.i465 = lshr i64 %0, 48
  %trunc = trunc i64 %shr.i465 to i15
  switch i15 %trunc, label %if.end26 [
    i15 -1, label %if.then14
    i15 0, label %if.then21
  ]

if.then14:                                        ; preds = %entry
  unreachable

if.then21:                                        ; preds = %entry
  unreachable

if.end26:                                         ; preds = %entry
  br label %if.end36

if.end36:                                         ; preds = %if.end26
  %or.i335 = or i64 undef, undef
  %shl2.i322 = or i64 undef, -9223372036854775808
  br i1 undef, label %if.then44, label %lor.rhs.i

lor.rhs.i:                                        ; preds = %if.end36
  br label %le128.exit

le128.exit:                                       ; preds = %lor.rhs.i
  br i1 undef, label %if.then44, label %while.cond.preheader

if.then44:                                        ; preds = %le128.exit, %if.end36
  %conv42544 = phi i64 [ 0, %le128.exit ], [ 1, %if.end36 ]
  br label %while.cond.preheader

while.cond.preheader:                             ; preds = %if.then44, %le128.exit
  %aSig0.3.ph = phi i64 [ undef, %if.then44 ], [ %or.i335, %le128.exit ]
  %q.0.ph = phi i64 [ %conv42544, %if.then44 ], [ 0, %le128.exit ]
  br i1 undef, label %while.body.lr.ph, label %while.end

while.body.lr.ph:                                 ; preds = %while.cond.preheader
  %shr.i263 = lshr i64 %shl2.i322, 32
  br label %while.body

while.body:                                       ; preds = %exit312, %while.body.lr.ph
  %aSig0.3554 = phi i64 [ %aSig0.3.ph, %while.body.lr.ph ], [ %sub3.i205, %exit312 ]
  br label %while.body.i297

while.body.i297:                                  ; preds = %while.body.i297, %while.body
  %z.045.i287 = phi i64 [ %sub.i290, %while.body.i297 ], [ undef, %while.body ]
  %sub.i290 = add i64 %z.045.i287, -4294967296
  %cmp3.i296 = icmp slt i64 undef, 0
  br i1 %cmp3.i296, label %while.body.i297, label %while.end.i305.loopexit

while.end.i305.loopexit:                          ; preds = %while.body.i297
  %or14.i309 = or i64 0, %sub.i290
  br label %exit312

exit312:                                          ; preds = %while.end.i305.loopexit
  %cmp50 = icmp ugt i64 %or14.i309, 4
  %cond = select i1 %cmp50, i64 undef, i64 0
  %shr3.i.i221 = lshr i64 %cond, 32
  %mul15.i11.i243 = mul nuw i64 %shr3.i.i221, %shr.i263
  %add20.i18.i250 = add i64 0, %mul15.i11.i243
  %add26.i23.i255 = add i64 %add20.i18.i250, 0
  %add3.i.i261 = add i64 %add26.i23.i255, 0
  %shl4.i215 = shl i64 %add3.i.i261, 61
  %or10.i = or i64 %shl4.i215, 0
  %shl2.i207 = shl i64 %aSig0.3554, 61
  %or.i209 = or i64 %shl2.i207, 0
  %sub1.i202 = add i64 0, %or.i209
  %sub3.i205 = sub i64 %sub1.i202, %or10.i
  %cmp47 = icmp sgt i32 undef, 61
  br i1 %cmp47, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %exit312
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %while.cond.preheader
  %aSig0.3.lcssa = phi i64 [ %aSig0.3.ph, %while.cond.preheader ], [ %sub3.i205, %while.end.loopexit ]
  %q.0.lcssa = phi i64 [ %q.0.ph, %while.cond.preheader ], [ %cond, %while.end.loopexit ]
  br i1 undef, label %if.then56, label %if.else71

if.then56:                                        ; preds = %while.end
  unreachable

if.else71:                                        ; preds = %while.end
  %shr8.i155 = lshr i64 %aSig0.3.lcssa, 12
  br label %do.body

do.body:                                          ; preds = %do.body, %if.else71
  %aSig0.5 = phi i64 [ %sub3.i151, %do.body ], [ %shr8.i155, %if.else71 ]
  %q.1 = phi i64 [ %inc, %do.body ], [ %q.0.lcssa, %if.else71 ]
  %inc = add i64 %q.1, 1
  %sub1.i148 = sub i64 %aSig0.5, 0
  %sub3.i151 = add i64 %sub1.i148, 0
  %cmp73 = icmp sgt i64 %sub3.i151, -1
  br i1 %cmp73, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  %and = and i64 %inc, 1
  unreachable
}

attributes #0 = { nounwind }
