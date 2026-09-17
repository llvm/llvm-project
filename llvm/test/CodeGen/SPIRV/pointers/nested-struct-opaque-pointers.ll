; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-PACKED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpTypeInt 8 0

; Deduction rebuilds this struct because the pointer member becomes a
; TypedPointerType. The rebuild must not drop packedness.
; CHECK-PACKED-DAG: OpDecorate %[[#PACKED:]] CPacked
; CHECK-PACKED-DAG: %[[#PACKED]] = OpTypeStruct %[[#]] %[[#]]

@GI = addrspace(1) constant i64 42

@GS = addrspace(1) global {ptr addrspace(1), ptr addrspace(1)} { ptr addrspace(1) @GI, ptr addrspace(1) @GI }
@GS2 = addrspace(1) global {ptr addrspace(1), ptr addrspace(1)} { ptr addrspace(1) @GS, ptr addrspace(1) @GS }
@GS3 = addrspace(1) global {ptr addrspace(1), ptr addrspace(1)} { ptr addrspace(1) @GS2, ptr addrspace(1) @GS2 }

@GPS = addrspace(1) global ptr addrspace(1) @GS3

@GPI1 = addrspace(1) global ptr addrspace(1) @GI
@GPI2 = addrspace(1) global ptr addrspace(1) @GPI1
@GPI3 = addrspace(1) global ptr addrspace(1) @GPI2

@GPACK1 = addrspace(1) global <{ i16, ptr addrspace(1) }> <{ i16 1, ptr addrspace(1) @GI }>
@GPACK2 = addrspace(1) global <{ i16, ptr addrspace(1) }> <{ i16 2, ptr addrspace(1) @GI }>

define spir_kernel void @foo() {
  ret void
}
