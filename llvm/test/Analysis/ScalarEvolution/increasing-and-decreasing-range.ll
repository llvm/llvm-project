; RUN: opt -analyze -scalar-evolution < %s | FileCheck %s

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
; CHECK-NEXT:  -->  {(-2 + %start)<nsw>,+,%step}<%loop> U: [-2,119) S: [-2,119)
; CHECK:  %iv.m3 = sub i32 %iv, 3
; CHECK-NEXT:  -->  {(-3 + %start)<nsw>,+,%step}<%loop> U: [-3,118) S: [-3,118)
; CHECK:  %iv.m4 = sub i32 %iv, 4
; CHECK-NEXT:  -->  {(-4 + %start)<nsw>,+,%step}<%loop> U: [-4,117) S: [-4,117)
; CHECK:  %iv.m5 = sub i32 %iv, 5
; CHECK-NEXT:  -->  {(-5 + %start)<nsw>,+,%step}<%loop> U: [-5,116) S: [-5,116)
; CHECK:  %iv.m6 = sub i32 %iv, 6
; CHECK-NEXT:  -->  {(-6 + %start)<nsw>,+,%step}<%loop> U: [-6,115) S: [-6,115)
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
