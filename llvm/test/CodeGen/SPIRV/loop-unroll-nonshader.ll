; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that OpLoopMerge is emitted for non-shader targets without requiring
; SPV_INTEL_unstructured_loop_controls.

; Test 1: llvm.loop.unroll.enable -> OpLoopMerge with Unroll.
; CHECK: OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll
; CHECK: OpBranchConditional
; CHECK: OpFunctionEnd

define spir_kernel void @test_unroll_enable(ptr addrspace(1) %dst) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !0

for.end:
  ret void
}

; Test 2: llvm.loop.unroll.disable -> OpLoopMerge with DontUnroll.
; CHECK: OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] DontUnroll
; CHECK: OpBranchConditional
; CHECK: OpFunctionEnd

define spir_kernel void @test_unroll_disable(ptr addrspace(1) %dst) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !1

for.end:
  ret void
}

; Test 3: llvm.loop.unroll.count N -> OpLoopMerge with PartialCount N.
; CHECK: OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] PartialCount 4
; CHECK: OpBranchConditional
; CHECK: OpFunctionEnd

define spir_kernel void @test_unroll_count(ptr addrspace(1) %dst) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !2

for.end:
  ret void
}

; Test 4: llvm.loop.unroll.full -> OpLoopMerge with Unroll.
; CHECK: OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll
; CHECK: OpBranchConditional
; CHECK: OpFunctionEnd

define spir_kernel void @test_unroll_full(ptr addrspace(1) %dst) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !3

for.end:
  ret void
}

; Test 5: Loop with multiple latches.
; The loop has two blocks branching back to the header; loop-simplify inserts
; a dedicated latch block so that getLoopLatch() returns a unique latch.
; CHECK: OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll
; CHECK: OpFunctionEnd

define spir_kernel void @test_multi_latch(ptr addrspace(1) %dst, i32 %flag) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch1 ], [ %inc, %latch2 ]
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  %cond = icmp eq i32 %flag, 0
  br i1 %cond, label %latch1, label %latch2

latch1:
  %cmp1 = icmp ult i32 %inc, 10
  br i1 %cmp1, label %header, label %exit, !llvm.loop !8

latch2:
  %cmp2 = icmp ult i32 %inc, 20
  br i1 %cmp2, label %header, label %exit, !llvm.loop !8

exit:
  ret void
}

; Test 6: Loop with multiple exits.
; The loop exits from two different blocks; loop-simplify inserts a dedicated
; exit block so that getUniqueExitBlock() succeeds.
; CHECK: OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] DontUnroll
; CHECK: OpFunctionEnd

define spir_kernel void @test_multi_exit(ptr addrspace(1) %dst, ptr addrspace(1) %cond_ptr) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %early_cond = load i32, ptr addrspace(1) %cond_ptr, align 4
  %early_exit = icmp eq i32 %early_cond, 42
  br i1 %early_exit, label %exit, label %latch

latch:
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %header, label %exit, !llvm.loop !9

exit:
  ret void
}

; Check that no Intel extension is required.
; CHECK-NOT: OpExtension "SPV_INTEL_unstructured_loop_controls"
; CHECK-NOT: OpCapability UnstructuredLoopControlsINTEL
; CHECK-NOT: OpLoopControlINTEL

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5}
!2 = distinct !{!2, !6}
!3 = distinct !{!3, !7}
!8 = distinct !{!8, !4}
!9 = distinct !{!9, !5}

!4 = !{!"llvm.loop.unroll.enable"}
!5 = !{!"llvm.loop.unroll.disable"}
!6 = !{!"llvm.loop.unroll.count", i32 4}
!7 = !{!"llvm.loop.unroll.full"}
