; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; Test case for optimizing AMDGPU ballot + assume patterns
; When we assume that ballot(cmp) == -1, we know that cmp is uniform
; This allows us to optimize away the ballot and directly branch

define void @test_assume_ballot_uniform(i32 %x) {
; CHECK-LABEL: @test_assume_ballot_uniform(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 true, label [[FOO:%.*]], label [[BAR:%.*]]
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

; Test case with partial optimization - only ballot removal without branch optimization
define void @test_assume_ballot_partial(i32 %x) {
; CHECK-LABEL: @test_assume_ballot_partial(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 true, label [[FOO:%.*]], label [[BAR:%.*]]
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

; Negative test - ballot not compared to -1
define void @test_assume_ballot_not_uniform(i32 %x) {
; CHECK-LABEL: @test_assume_ballot_not_uniform(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 0
; CHECK-NEXT:    [[BALLOT:%.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP]])
; CHECK-NEXT:    [[SOME:%.*]] = icmp ne i64 [[BALLOT]], 0
; CHECK-NEXT:    call void @llvm.assume(i1 [[SOME]])
; CHECK-NEXT:    br i1 [[CMP]], label [[FOO:%.*]], label [[BAR:%.*]]
; CHECK:       foo:
; CHECK-NEXT:    ret void
; CHECK:       bar:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq i32 %x, 0
  %ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %cmp)
  %some = icmp ne i64 %ballot, 0
  call void @llvm.assume(i1 %some)
  br i1 %cmp, label %foo, label %bar

foo:
  ret void

bar:
  ret void
}

; Test with 32-bit ballot
define void @test_assume_ballot_uniform_i32(i32 %x) {
; CHECK-LABEL: @test_assume_ballot_uniform_i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 true, label [[FOO:%.*]], label [[BAR:%.*]]
; CHECK:       foo:
; CHECK-NEXT:    ret void
; CHECK:       bar:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq i32 %x, 0
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 %cmp)
  %all = icmp eq i32 %ballot, -1  
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
