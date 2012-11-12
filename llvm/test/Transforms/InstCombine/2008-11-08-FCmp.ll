; RUN: opt < %s -instcombine -S | FileCheck %s
; PR3021

; When inst combining an FCMP with the LHS coming from a uitofp instruction, we
; can't lower it to signed ICMP instructions.

define i1 @test1(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp ole double %1, 0.000000e+00
; CHECK: icmp eq i32 %val, 0
  ret i1 %2
}

define i1 @test2(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp olt double %1, 0.000000e+00
  ret i1 %2
; CHECK: ret i1 false
}

define i1 @test3(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp oge double %1, 0.000000e+00
  ret i1 %2
; CHECK: ret i1 true
}

define i1 @test4(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp ogt double %1, 0.000000e+00
; CHECK: icmp ne i32 %val, 0
  ret i1 %2
}

define i1 @test5(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp ogt double %1, -4.400000e+00
  ret i1 %2
; CHECK: ret i1 true
}

define i1 @test6(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp olt double %1, -4.400000e+00
  ret i1 %2
; CHECK: ret i1 false
}

; Check that optimizing unsigned >= comparisons correctly distinguishes
; positive and negative constants.  <rdar://problem/12029145>
define i1 @test7(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp oge double %1, 3.200000e+00
  ret i1 %2
; CHECK: icmp ugt i32 %val, 3
}
