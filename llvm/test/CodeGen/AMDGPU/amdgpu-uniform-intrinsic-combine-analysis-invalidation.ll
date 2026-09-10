; REQUIRES: asserts
; RUN: opt -mtriple=amdgpu10.10-amd-amdhsa -passes=amdgpu-uniform-intrinsic-combine -S < %s | FileCheck %s
; RUN: opt -mtriple=amdgpu10.10-amd-amdhsa -passes='amdgpu-uniform-intrinsic-combine,reassociate' -verify-analysis-invalidation -disable-output < %s

; Reassociate must not use uniformity information that refers to invalidated
; cycle information after the combine changes the IR.

; The pass must report a change when erasing an already unused ballot.
define amdgpu_kernel void @erase_unused_ballot() {
; CHECK-LABEL: define amdgpu_kernel void @erase_unused_ballot()
; CHECK-NOT: call i64 @llvm.amdgcn.ballot
; CHECK: ret void
entry:
  %mask = call i64 @llvm.amdgcn.ballot.i64(i1 false)
  ret void
}

define amdgpu_kernel void @invalidate_uniformity_info() {
; CHECK-LABEL: define amdgpu_kernel void @invalidate_uniformity_info()
; CHECK: ballot:
; CHECK-NEXT: %[[MASK:.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 false)
; CHECK-NEXT: %[[NOT:.*]] = xor i1 false, true
; CHECK-NEXT: %none.active = icmp eq i64 %[[MASK]], 0
entry:
  %base = mul i32 0, 0
  %factor = zext i16 0 to i32
  %product = mul i32 %base, %factor
  %sum = or i32 0, %product
  br label %dispatch

dispatch:
  br label %select

select:
  switch i32 0, label %default [
    i32 0, label %ballot
    i32 1, label %left
    i32 2, label %right
  ]

ballot:
  %mask = call i64 @llvm.amdgcn.ballot.i64(i1 false)
  %none.active = icmp eq i64 %mask, 0
  br label %unreachable.block

unreachable.block:
  unreachable

left:
  br label %left.next

left.next:
  br label %left.merge

left.merge:
  br label %exit

right:
  br label %right.next

right.next:
  br label %right.merge

right.merge:
  br label %exit

default:
  ret void

exit:
  ret void
}
