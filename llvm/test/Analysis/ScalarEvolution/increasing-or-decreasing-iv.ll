; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define void @f0(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f0
entry:
  %start = select i1 %c, i32 127, i32 0
  %step  = select i1 %c, i32 -1,  i32 1
  br label %loop

loop:
  %loop.iv = phi i32 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
; CHECK: %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:  -->  {%start,+,%step}<%loop> U: [0,128) S: [0,128)
  %iv.next = add i32 %iv, %step
  %loop.iv.inc = add i32 %loop.iv, 1
  %be.cond = icmp ne i32 %loop.iv.inc, 128
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}

define void @f1(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f1
entry:
  %start = select i1 %c, i32 120, i32 0
  %step  = select i1 %c, i32 -8,  i32 8
  br label %loop

loop:
  %loop.iv = phi i32 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]

; CHECK:  %iv.1 = add i32 %iv, 1
; CHECK-NEXT:  -->  {(1 + %start)<nuw><nsw>,+,%step}<%loop> U: [1,122) S: [1,122)
; CHECK:  %iv.2 = add i32 %iv, 2
; CHECK-NEXT:  -->  {(2 + %start)<nuw><nsw>,+,%step}<%loop> U: [2,123) S: [2,123)
; CHECK:  %iv.3 = add i32 %iv, 3
; CHECK-NEXT:  -->  {(3 + %start)<nuw><nsw>,+,%step}<%loop> U: [3,124) S: [3,124)
; CHECK:  %iv.4 = add i32 %iv, 4
; CHECK-NEXT:  -->  {(4 + %start)<nuw><nsw>,+,%step}<%loop> U: [4,125) S: [4,125)
; CHECK:  %iv.5 = add i32 %iv, 5
; CHECK-NEXT:  -->  {(5 + %start)<nuw><nsw>,+,%step}<%loop> U: [5,126) S: [5,126)
; CHECK:  %iv.6 = add i32 %iv, 6
; CHECK-NEXT:  -->  {(6 + %start)<nuw><nsw>,+,%step}<%loop> U: [6,127) S: [6,127)
; CHECK:  %iv.7 = add i32 %iv, 7
; CHECK-NEXT:  -->  {(7 + %start)<nuw><nsw>,+,%step}<%loop> U: [7,128) S: [7,128)

  %iv.1 = add i32 %iv, 1
  %iv.2 = add i32 %iv, 2
  %iv.3 = add i32 %iv, 3
  %iv.4 = add i32 %iv, 4
  %iv.5 = add i32 %iv, 5
  %iv.6 = add i32 %iv, 6
  %iv.7 = add i32 %iv, 7

; CHECK:  %iv.m1 = sub i32 %iv, 1
; CHECK-NEXT:  -->  {(-1 + %start)<nsw>,+,%step}<%loop> U: [-1,120) S: [-1,120)
; CHECK:  %iv.m2 = sub i32 %iv, 2
; CHECK-NEXT:  -->  {(-2 + %start)<nsw>,+,%step}<%loop> U: [0,-1) S: [-2,119)
; CHECK:  %iv.m3 = sub i32 %iv, 3
; CHECK-NEXT:  -->  {(-3 + %start)<nsw>,+,%step}<%loop> U: [-3,118) S: [-3,118)
; CHECK:  %iv.m4 = sub i32 %iv, 4
; CHECK-NEXT:  -->  {(-4 + %start)<nsw>,+,%step}<%loop> U: [0,-3) S: [-4,117)
; CHECK:  %iv.m5 = sub i32 %iv, 5
; CHECK-NEXT:  -->  {(-5 + %start)<nsw>,+,%step}<%loop> U: [-5,116) S: [-5,116)
; CHECK:  %iv.m6 = sub i32 %iv, 6
; CHECK-NEXT:  -->  {(-6 + %start)<nsw>,+,%step}<%loop> U: [0,-1) S: [-6,115)
; CHECK:  %iv.m7 = sub i32 %iv, 7
; CHECK-NEXT:  -->  {(-7 + %start)<nsw>,+,%step}<%loop> U: [-7,114) S: [-7,114)

  %iv.m1 = sub i32 %iv, 1
  %iv.m2 = sub i32 %iv, 2
  %iv.m3 = sub i32 %iv, 3
  %iv.m4 = sub i32 %iv, 4
  %iv.m5 = sub i32 %iv, 5
  %iv.m6 = sub i32 %iv, 6
  %iv.m7 = sub i32 %iv, 7

  %iv.next = add i32 %iv, %step
  %loop.iv.inc = add i32 %loop.iv, 1
  %be.cond = icmp sgt i32 %loop.iv, 14
  br i1 %be.cond, label %leave, label %loop

leave:
  ret void
}

define void @f2(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f2
entry:
  %start = select i1 %c, i32 127, i32 0
  %step  = select i1 %c, i32 -1,  i32 1
  br label %loop

loop:
  %loop.iv = phi i32 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
  %iv.sext = sext i32 %iv to i64
  %iv.next = add i32 %iv, %step
; CHECK:  %iv.sext = sext i32 %iv to i64
; CHECK-NEXT:   -->  {(sext i32 %start to i64),+,(sext i32 %step to i64)}<nsw><%loop> U: [0,128) S: [0,128)
  %loop.iv.inc = add i32 %loop.iv, 1
  %be.cond = icmp ne i32 %loop.iv.inc, 128
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}

define void @f3(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f3
entry:

; NB! the i16 type (as opposed to i32), the choice of the constant 509
; and the trip count are all related and not arbitrary.  We want an
; add recurrence that will look like it can unsign-overflow *unless*
; SCEV is able to see the correlation between the two selects feeding
; into the initial value and the step increment.

  %start = select i1 %c, i16 1000, i16 0
  %step  = select i1 %c, i16 1,  i16 509
  br label %loop

loop:
  %loop.iv = phi i16 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i16 [ %start, %entry ], [ %iv.next, %loop ]
  %iv.zext = zext i16 %iv to i64
; CHECK:  %iv.zext = zext i16 %iv to i64
; CHECK-NEXT:  -->  {(zext i16 %start to i64),+,(zext i16 %step to i64)}<nuw><%loop> U: [0,64644) S: [0,64644)
  %iv.next = add i16 %iv, %step
  %loop.iv.inc = add i16 %loop.iv, 1
  %be.cond = icmp ne i16 %loop.iv.inc, 128
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}

define void @f4(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f4

; @f4() demonstrates a case where SCEV is not able to compute a
; precise range for %iv.trunc, though it should be able to, in theory.
; This is because SCEV looks into affine add recurrences only when the
; backedge taken count of the loop has the same bitwidth as the
; induction variable.
entry:
  %start = select i1 %c, i32 127, i32 0
  %step  = select i1 %c, i32 -1,  i32 1
  br label %loop

loop:
  %loop.iv = phi i32 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
  %iv.trunc = trunc i32 %iv to i16
; CHECK:  %iv.trunc = trunc i32 %iv to i16
; CHECK-NEXT:   -->  {(trunc i32 %start to i16),+,(trunc i32 %step to i16)}<%loop> U: full-set S: full-set
  %iv.next = add i32 %iv, %step
  %loop.iv.inc = add i32 %loop.iv, 1
  %be.cond = icmp ne i32 %loop.iv.inc, 128
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}

define void @f5(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f5
entry:
  %start = select i1 %c, i32 127, i32 0
  %step  = select i1 %c, i32 -1,  i32 1
  br label %loop

loop:
  %loop.iv = phi i16 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
  %iv.trunc = trunc i32 %iv to i16
; CHECK:  %iv.trunc = trunc i32 %iv to i16
; CHECK-NEXT:  -->  {(trunc i32 %start to i16),+,(trunc i32 %step to i16)}<%loop> U: [0,128) S: [0,128)
  %iv.next = add i32 %iv, %step

  %loop.iv.inc = add i16 %loop.iv, 1
  %be.cond = icmp ne i16 %loop.iv.inc, 128
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}

define void @f6(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f6
entry:
  %start = select i1 %c, i32 127, i32 0
  %step  = select i1 %c, i32 -2,  i32 0
  br label %loop

loop:
  %loop.iv = phi i16 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
; CHECK:   %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:  -->  {%start,+,(1 + %step)<nuw><nsw>}<%loop> U: [0,128) S: [0,128)

  %step.plus.one = add i32 %step, 1
  %iv.next = add i32 %iv, %step.plus.one
  %iv.sext = sext i32 %iv to i64
; CHECK:   %iv.sext = sext i32 %iv to i64
; CHECK-NEXT:  -->  {(sext i32 %start to i64),+,(1 + (sext i32 %step to i64))<nuw><nsw>}<nsw><%loop> U: [0,128) S: [0,128)
  %loop.iv.inc = add i16 %loop.iv, 1
  %be.cond = icmp ne i16 %loop.iv.inc, 128
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}

define void @f7(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f7
entry:
  %start = select i1 %c, i32 127, i32 0
  %step  = select i1 %c, i32 -1,  i32 1
  br label %loop

loop:
  %loop.iv = phi i16 [ 0, %entry ], [ %loop.iv.inc, %loop ]
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop ]
  %iv.trunc = trunc i32 %iv to i16
; CHECK:  %iv.trunc = trunc i32 %iv to i16
; CHECK-NEXT:  -->  {(trunc i32 %start to i16),+,(trunc i32 %step to i16)}<%loop> U: [0,128) S: [0,128)
  %iv.next = add i32 %iv, %step

  %iv.trunc.plus.one = add i16 %iv.trunc, 1
; CHECK:  %iv.trunc.plus.one = add i16 %iv.trunc, 1
; CHECK-NEXT:  -->  {(1 + (trunc i32 %start to i16))<nuw><nsw>,+,(trunc i32 %step to i16)}<%loop> U: [1,129) S: [1,129)

  %iv.trunc.plus.two = add i16 %iv.trunc, 2
; CHECK:  %iv.trunc.plus.two = add i16 %iv.trunc, 2
; CHECK-NEXT:  -->  {(2 + (trunc i32 %start to i16))<nuw><nsw>,+,(trunc i32 %step to i16)}<%loop> U: [2,130) S: [2,130)

  %loop.iv.inc = add i16 %loop.iv, 1
  %be.cond = icmp ne i16 %loop.iv.inc, 128
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}
