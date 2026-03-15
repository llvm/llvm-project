; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_unstructured_loop_controls %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NO-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_unstructured_loop_controls %s -o - -filetype=obj | spirv-val %}

; Check that extension and capability are emitted when extension is enabled.
; CHECK-SPIRV-DAG: OpCapability UnstructuredLoopControlsINTEL
; CHECK-SPIRV-DAG: OpExtension "SPV_INTEL_unstructured_loop_controls"

; Check that OpLoopControlINTEL is NOT emitted when extension is not enabled.
; CHECK-NO-EXT-NOT: OpLoopControlINTEL

; Test 1: llvm.loop.unroll.enable -> OpLoopControlINTEL Unroll.
; CHECK-SPIRV: test_unroll_enable
; CHECK-SPIRV: OpLoopControlINTEL Unroll
; CHECK-SPIRV-NEXT: OpBranchConditional

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

; Test 2: llvm.loop.unroll.disable -> OpLoopControlINTEL DontUnroll.
; CHECK-SPIRV: test_unroll_disable
; CHECK-SPIRV: OpLoopControlINTEL DontUnroll
; CHECK-SPIRV-NEXT: OpBranchConditional

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

; Test 3: llvm.loop.unroll.count N -> OpLoopControlINTEL PartialCount N.
; CHECK-SPIRV: test_unroll_count
; CHECK-SPIRV: OpLoopControlINTEL PartialCount 4
; CHECK-SPIRV-NEXT: OpBranchConditional

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

; Test 4: llvm.loop.unroll.full -> OpLoopControlINTEL Unroll.
; CHECK-SPIRV: test_unroll_full
; CHECK-SPIRV: OpLoopControlINTEL Unroll
; CHECK-SPIRV-NEXT: OpBranchConditional

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

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5}
!2 = distinct !{!2, !6}
!3 = distinct !{!3, !7}

!4 = !{!"llvm.loop.unroll.enable"}
!5 = !{!"llvm.loop.unroll.disable"}
!6 = !{!"llvm.loop.unroll.count", i32 4}
!7 = !{!"llvm.loop.unroll.full"}
