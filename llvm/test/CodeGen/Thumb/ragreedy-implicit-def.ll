; REQUIRES: asserts
; RUN: llc -mtriple=thumbv6m -regalloc=greedy -stats < %s 2>&1 | FileCheck %s

; Undef incoming values to phis end up creating IMPLICIT_DEF values. If we don't
; prefer them to be in a register then we get fewer spilled live ranges (6
; compared to 7).
; CHECK: 6 regalloc - Number of spilled live ranges

declare i32 @otherfn(i32)
define void @fn(i32 %val, ptr %ptr) {
entry:
  %gep2 = getelementptr i32, ptr %ptr, i32 1
  %gep3 = getelementptr i32, ptr %ptr, i32 2
  %gep4 = getelementptr i32, ptr %ptr, i32 3
  %gep5 = getelementptr i32, ptr %ptr, i32 4
  %gep6 = getelementptr i32, ptr %ptr, i32 5
  %gep7 = getelementptr i32, ptr %ptr, i32 6
  %gep8 = getelementptr i32, ptr %ptr, i32 7
  %cmp1 = icmp uge i32 %val, 3
  br i1 %cmp1, label %if, label %then

if:
  %val1 = load i32, ptr %ptr, align 4
  %val2 = load i32, ptr %gep2, align 4
  %val3 = load i32, ptr %gep3, align 4
  %val4 = load i32, ptr %gep4, align 4
  %val5 = load i32, ptr %gep5, align 4
  %val6 = load i32, ptr %gep6, align 4
  %val7 = load i32, ptr %gep7, align 4
  %val8 = load i32, ptr %gep8, align 4
  br label %then

then:
  %phi1a = phi i32 [ %val1, %if ], [ undef, %entry ]
  %phi2a = phi i32 [ %val2, %if ], [ undef, %entry ]
  %phi3a = phi i32 [ %val3, %if ], [ undef, %entry ]
  %phi4a = phi i32 [ %val4, %if ], [ undef, %entry ]
  %phi5a = phi i32 [ %val5, %if ], [ undef, %entry ]
  %phi6a = phi i32 [ %val6, %if ], [ undef, %entry ]
  %phi7a = phi i32 [ %val7, %if ], [ undef, %entry ]
  %phi8a = phi i32 [ %val8, %if ], [ undef, %entry ]
  %switchval = call i32 @otherfn(i32 %val)
  switch i32 %switchval, label %default [
    i32 0, label %case0
    i32 1, label %case1
    i32 5, label %case5
    i32 6, label %case6
    i32 7, label %case7
    i32 8, label %case8
  ]

default:
  br label %switchend

case0:
  br label %switchend

case1:
  br label %switchend

case5:
  br label %switchend

case6:
  br label %switchend

case7:
  br label %switchend

case8:
  br label %switchend

switchend:
  %phi1b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi1a, %case5 ], [ 1, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %phi2b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi2a, %case5 ], [ 2, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %phi3b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi3a, %case5 ], [ 3, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %phi4b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi4a, %case5 ], [ 4, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %phi5b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi5a, %case5 ], [ 5, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %phi6b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi6a, %case5 ], [ 6, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %phi7b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi7a, %case5 ], [ 7, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %phi8b = phi i32 [ 0, %default ], [ undef, %case0 ], [ undef, %case1 ], [ %phi8a, %case5 ], [ 8, %case6 ], [ 2, %case7 ], [ 1, %case8 ]
  %cmp2 = icmp uge i32 %val, 4
  br i1 %cmp2, label %if2, label %end

if2:
  store i32 %phi1b, ptr %ptr, align 4
  store i32 %phi2b, ptr %gep2, align 4
  store i32 %phi3b, ptr %gep3, align 4
  store i32 %phi4b, ptr %gep4, align 4
  store i32 %phi5b, ptr %gep5, align 4
  store i32 %phi6b, ptr %gep6, align 4
  store i32 %phi7b, ptr %gep7, align 4
  store i32 %phi8b, ptr %gep8, align 4
  br label %end

end:
  ret void
}
