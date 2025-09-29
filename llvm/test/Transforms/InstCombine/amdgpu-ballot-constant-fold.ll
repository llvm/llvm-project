; RUN: opt < %s -mtriple=amdgcn-amd-amdhsa -passes=instcombine -S | FileCheck %s

; Test cases for optimizing AMDGPU ballot intrinsics
; Focus on constant folding ballot(false) -> 0 and poison handling

; Test ballot with constant false condition gets folded
define i32 @test_ballot_constant_false() {
; CHECK-LABEL: @test_ballot_constant_false(
; CHECK-NEXT:    ret i32 0
;
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 false)
  ret i32 %ballot
}

; Test ballot.i64 with constant false condition gets folded
define i64 @test_ballot_i64_constant_false() {
; CHECK-LABEL: @test_ballot_i64_constant_false(
; CHECK-NEXT:    ret i64 0
;
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 false)
  ret i64 %ballot
}

; Test ballot with poison condition gets folded to poison
define i64 @test_ballot_poison() {
; CHECK-LABEL: @test_ballot_poison(
; CHECK-NEXT:    ret i64 poison
;
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 poison)
  ret i64 %ballot
}

; Test that ballot(true) is NOT constant folded (depends on active lanes)
define i64 @test_ballot_constant_true() {
; CHECK-LABEL: @test_ballot_constant_true(
; CHECK-NEXT:    [[BALLOT:%.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 true)
; CHECK-NEXT:    ret i64 [[BALLOT]]
;
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 true)
  ret i64 %ballot
}

; Test that ballot with variable condition is not optimized
define i64 @test_ballot_variable_condition(i32 %x) {
; CHECK-LABEL: @test_ballot_variable_condition(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 0
; CHECK-NEXT:    [[BALLOT:%.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; CHECK-NEXT:    ret i64 [[BALLOT]]
;
  %cmp = icmp eq i32 %x, 0
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %cmp)
  ret i64 %ballot
}

declare i64 @llvm.amdgcn.ballot.i64(i1)
declare i32 @llvm.amdgcn.ballot.i32(i1)
