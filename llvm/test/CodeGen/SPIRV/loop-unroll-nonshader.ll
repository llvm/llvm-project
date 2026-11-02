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

; Test 7: Loop header ending in a switch.
; OpLoopMerge must immediately precede OpBranch or OpBranchConditional, so no
; OpLoopMerge may be emitted when the loop header terminates in OpSwitch.
; CHECK: OpFunction
; CHECK-NOT: OpLoopMerge
; CHECK: OpSwitch
; CHECK-NOT: OpLoopMerge
; CHECK: OpFunctionEnd

define spir_kernel void @test_switch_header(ptr addrspace(1) %dst) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  switch i32 %i, label %exit [
    i32 0, label %latch
    i32 1, label %latch
  ]

latch:
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  br label %header, !llvm.loop !0

exit:
  ret void
}

; Test 8: Loop with multiple distinct exit blocks.
; The loop exits to two different blocks (%early.out and %latch.out), so
; getUniqueExitBlock() returns nullptr; the backend falls back to the latch's
; exit successor (%latch.out) as the OpLoopMerge merge block.
; CHECK: OpFunction
; CHECK: OpLoopMerge %[[#MERGE:]] %[[#LATCH:]] DontUnroll
; CHECK-NEXT: OpBranchConditional %[[#]] %[[#]] %[[#LATCH]]
; CHECK: %[[#LATCH]] = OpLabel
; CHECK: OpBranchConditional %[[#]] %[[#]] %[[#MERGE]]
; CHECK: %[[#MERGE]] = OpLabel
; CHECK: OpFunctionEnd

define spir_kernel void @test_multi_distinct_exits(ptr addrspace(1) %dst, ptr addrspace(1) %cond_ptr) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %early_cond = load i32, ptr addrspace(1) %cond_ptr, align 4
  %early_exit = icmp eq i32 %early_cond, 42
  br i1 %early_exit, label %early.out, label %latch

latch:
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %header, label %latch.out, !llvm.loop !9

early.out:
  store i32 -1, ptr addrspace(1) %dst, align 4
  ret void

latch.out:
  ret void
}

; Test 9: Multiple distinct exits with an unconditional latch.
; Use the header's exit successor as the merge block.
; CHECK: OpFunction
; CHECK: OpBranch %[[#HEADER:]]
; CHECK: %[[#HEADER]] = OpLabel
; CHECK: OpLoopMerge %[[#HEADER_MERGE:]] %[[#HEADER_LATCH:]] DontUnroll
; CHECK-NEXT: OpBranchConditional %[[#]] %[[#HEADER_MERGE]] %[[#]]
; CHECK: %[[#HEADER_LATCH]] = OpLabel
; CHECK: OpBranch %[[#HEADER]]
; CHECK: %[[#HEADER_MERGE]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK: OpFunctionEnd

define spir_kernel void @test_multi_exit_header(ptr addrspace(1) %dst, i32 %n, i1 %early) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %done = icmp uge i32 %i, %n
  br i1 %done, label %normal.out, label %body

body:
  br i1 %early, label %early.out, label %latch

latch:
  store i32 %i, ptr addrspace(1) %dst, align 4
  %inc = add i32 %i, 1
  br label %header, !llvm.loop !9

normal.out:
  ret void

early.out:
  store i32 -1, ptr addrspace(1) %dst, align 4
  ret void
}

; Test 10: A loop with a switch header and multiple distinct exit blocks.
; Skip OpLoopMerge even though the latch has an exit successor: it cannot
; precede OpSwitch, and emitting it after OpSwitch would place it outside a block.
; CHECK: OpFunction
; CHECK-NOT: OpLoopMerge
; CHECK: OpSwitch
; CHECK-NOT: OpLoopMerge
; CHECK: OpFunctionEnd

define spir_kernel void @test_multi_exit_switch_header(ptr addrspace(1) %dst, ptr addrspace(1) %cond_ptr) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %cond = load i32, ptr addrspace(1) %cond_ptr, align 4
  switch i32 %cond, label %latch [
    i32 42, label %early.out
    i32 43, label %latch
  ]

latch:
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %dst, i32 %i
  store i32 %i, ptr addrspace(1) %ptr, align 4
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %inc, 10
  br i1 %cmp, label %header, label %latch.out, !llvm.loop !9

early.out:
  store i32 -1, ptr addrspace(1) %dst, align 4
  ret void

latch.out:
  ret void
}

; Test 11: Nested loops must not share a merge block.
; The outer loop uses %done. The inner loop has two exits, but its header's
; exit is also %done, so omit its hint rather than reuse the outer merge block.
; CHECK: OpFunction
; CHECK-NOT: OpLoopMerge
; CHECK: OpLoopMerge %[[#OUTER_MERGE:]] %[[#]] Unroll
; CHECK-NOT: OpLoopMerge
; CHECK: %[[#OUTER_MERGE]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NOT: OpLoopMerge
; CHECK: OpFunctionEnd

define spir_kernel void @nested_shared_merge(i1 %c, i1 %d) {
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  br i1 %c, label %done, label %inner.body

inner.body:
  br i1 %d, label %inner.exit, label %inner.latch

inner.latch:
  br label %inner.header, !llvm.loop !9

inner.exit:
  br label %outer.latch

outer.latch:
  br label %outer.header, !llvm.loop !0

done:
  ret void
}

; Test 12: Multiple exits from body blocks only.
; Neither the header nor the latch has an exit successor. Leave the hint
; unlowered rather than choose an arbitrary body exit as the merge block.
; CHECK: OpFunction
; CHECK-NOT: OpLoopMerge
; CHECK: OpFunctionEnd

define spir_kernel void @test_multi_exit_body(ptr addrspace(1) %dst, i1 %a, i1 %b) {
entry:
  br label %header

header:
  store i32 0, ptr addrspace(1) %dst, align 4
  br label %body

body:
  br i1 %a, label %normal.out, label %body2

body2:
  br i1 %b, label %early.out, label %latch

latch:
  br label %header, !llvm.loop !9

normal.out:
  ret void

early.out:
  store i32 1, ptr addrspace(1) %dst, align 4
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
