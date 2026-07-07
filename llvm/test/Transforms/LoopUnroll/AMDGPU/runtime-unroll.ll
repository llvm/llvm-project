; RUN: opt -mtriple=amdgcn-- -passes=loop-unroll -S %s | FileCheck %s

; Verify that AMDGPU enables runtime loop unrolling for loops whose trip
; count is not known at compile time.

; Simple loop with unknown trip count — should be runtime-unrolled.
define void @runtime_unroll_simple(ptr addrspace(1) %out, i32 %n) {
; CHECK-LABEL: @runtime_unroll_simple(
; CHECK: %xtraiter = and i32 %n,
; CHECK: for.body.epil.preheader:
; CHECK: for.body.epil:
; CHECK: %epil.iter
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %idx = zext i32 %iv to i64
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx
  store i32 %iv, ptr addrspace(1) %ptr, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; Loop with convergent call — runtime unrolling must NOT fire because the
; prologue/epilogue would introduce divergent control flow around the
; convergent operation.
define void @no_runtime_unroll_convergent(ptr addrspace(1) %out, i32 %n) {
; CHECK-LABEL: @no_runtime_unroll_convergent(
; CHECK-NOT: xtraiter
; CHECK-NOT: epil
; CHECK: for.body:
; CHECK: %iv = phi i32
; CHECK: call i32 @llvm.amdgcn.readfirstlane.i32
; CHECK: %iv.next = add nuw nsw i32 %iv, 1
; CHECK: %exitcond = icmp eq i32 %iv.next, %n
; CHECK: br i1 %exitcond
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %lane0 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %iv)
  %idx = zext i32 %lane0 to i64
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx
  store i32 %iv, ptr addrspace(1) %ptr, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #0

attributes #0 = { nounwind convergent willreturn readnone }
