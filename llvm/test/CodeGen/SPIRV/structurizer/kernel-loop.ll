; RUN: llc -mtriple=spirv64-unknown-opencl -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-opencl %s -o - -filetype=obj | spirv-val %}

; This test verifies that the structurizer pass runs for OpenCL kernels,
; generating structured control flow with OpLoopMerge instructions and
; translating loop metadata to appropriate LoopControl operands.

; CHECK: OpEntryPoint Kernel %[[#kernel_unroll:]] "test_kernel_unroll"
; CHECK: OpEntryPoint Kernel %[[#kernel_dontunroll:]] "test_kernel_dontunroll"

; Verify unroll metadata is translated to Unroll LoopControl
; CHECK: %[[#kernel_unroll]] = OpFunction
; CHECK: OpLabel
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll
; CHECK: OpFunctionEnd

; Verify dont_unroll metadata is translated to DontUnroll LoopControl
; CHECK: %[[#kernel_dontunroll]] = OpFunction
; CHECK: OpLabel
; CHECK: OpLoopMerge %[[#]] %[[#]] DontUnroll
; CHECK: OpFunctionEnd

define spir_kernel void @test_kernel_unroll(ptr addrspace(1) %out) {
entry:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %1 = load i32, ptr %i, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %out, i64 0
  store i32 %1, ptr addrspace(1) %arrayidx, align 4
  br label %for.inc

for.inc:
  %2 = load i32, ptr %i, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !0

for.end:
  ret void
}

define spir_kernel void @test_kernel_dontunroll(ptr addrspace(1) %out) {
entry:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %1 = load i32, ptr %i, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %out, i64 0
  store i32 %1, ptr addrspace(1) %arrayidx, align 4
  br label %for.inc

for.inc:
  %2 = load i32, ptr %i, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !2

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.disable"}
