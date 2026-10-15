; AMD triples behave as if -spirv-preserve-auxdata were passed: function
; attributes such as convergent are preserved without the flag.

; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - \
; RUN:   -filetype=obj | spirv-val %}

; Without SPV_KHR_relaxed_extended_instruction, function records are dropped.
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_relaxed_extended_instruction %s -o - \
; RUN:   | FileCheck %s --check-prefix=NORELAXED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_relaxed_extended_instruction %s -o - \
; RUN:   -filetype=obj | spirv-val %}

; NORELAXED-NOT: SPV_KHR_relaxed_extended_instruction
; NORELAXED-NOT: OpString "convergent"
; NORELAXED-NOT: OpExtInstWithForwardRefsKHR

; CHECK-DAG: OpExtension "SPV_KHR_relaxed_extended_instruction"
; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#convergent:]] = OpString "convergent"
; CHECK-DAG: OpName %[[#fn:]] "my_barrier"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#fn]] %[[#convergent]]

declare void @llvm.amdgcn.s.barrier() convergent

define void @my_barrier() convergent {
  call void @llvm.amdgcn.s.barrier()
  ret void
}

define amdgpu_kernel void @k() {
  call void @my_barrier()
  ret void
}
