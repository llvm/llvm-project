; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; Test cases for optimizing AMDGPU ballot intrinsics
; Focus on constant folding ballot(true) -> -1 and ballot(false) -> 0

define void @test_ballot_constant_true() {
; CHECK-LABEL: @test_ballot_constant_true(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ALL:%.*]] = icmp eq i64 -1, -1
; CHECK-NEXT:    call void @llvm.assume(i1 [[ALL]])
; CHECK-NEXT:    br i1 true, label [[FOO:%.*]], label [[BAR:%.*]]
; CHECK:       foo:
; CHECK-NEXT:    ret void
; CHECK:       bar:
; CHECK-NEXT:    ret void
;
entry:
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 true)
  %all = icmp eq i64 %ballot, -1
  call void @llvm.assume(i1 %all)
  br i1 true, label %foo, label %bar

foo:
  ret void

bar:
  ret void
}

define void @test_ballot_constant_false() {
; CHECK-LABEL: @test_ballot_constant_false(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[NONE:%.*]] = icmp ne i64 0, 0
; CHECK-NEXT:    call void @llvm.assume(i1 [[NONE]])
; CHECK-NEXT:    br i1 false, label [[FOO:%.*]], label [[BAR:%.*]]
; CHECK:       foo:
; CHECK-NEXT:    ret void
; CHECK:       bar:
; CHECK-NEXT:    ret void
;
entry:
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 false)
  %none = icmp ne i64 %ballot, 0
  call void @llvm.assume(i1 %none)
  br i1 false, label %foo, label %bar

foo:
  ret void

bar:
  ret void
}

; Test with 32-bit ballot constants
define void @test_ballot_i32_constant_true() {
; CHECK-LABEL: @test_ballot_i32_constant_true(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ALL:%.*]] = icmp eq i32 -1, -1
; CHECK-NEXT:    call void @llvm.assume(i1 [[ALL]])
; CHECK-NEXT:    br i1 true, label [[FOO:%.*]], label [[BAR:%.*]]
; CHECK:       foo:
; CHECK-NEXT:    ret void
; CHECK:       bar:
; CHECK-NEXT:    ret void
;
entry:
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  %all = icmp eq i32 %ballot, -1
  call void @llvm.assume(i1 %all)
  br i1 true, label %foo, label %bar

foo:
  ret void

bar:
  ret void
}

; Negative test - variable condition should not be optimized
define void @test_ballot_variable_condition(i32 %x) {
; CHECK-LABEL: @test_ballot_variable_condition(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 0
; CHECK-NEXT:    [[BALLOT:%.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; CHECK-NEXT:    [[ALL:%.*]] = icmp eq i64 [[BALLOT]], -1
; CHECK-NEXT:    call void @llvm.assume(i1 [[ALL]])
; CHECK-NEXT:    br i1 [[CMP]], label [[FOO:%.*]], label [[BAR:%.*]]
; CHECK:       foo:
; CHECK-NEXT:    ret void
; CHECK:       bar:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq i32 %x, 0
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %cmp)
  %all = icmp eq i64 %ballot, -1
  call void @llvm.assume(i1 %all)
  br i1 %cmp, label %foo, label %bar

foo:
  ret void

bar:
  ret void
}

declare i64 @llvm.amdgcn.ballot.i64(i1)
declare i32 @llvm.amdgcn.ballot.i32(i1)
declare void @llvm.assume(i1)
