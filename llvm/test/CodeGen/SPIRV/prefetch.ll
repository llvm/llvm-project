; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck --check-prefixes=NO-EXT %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck --check-prefixes=EXT %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; NO-EXT: %[[#EXTSET:]] = OpExtInstImport "OpenCL.std"

define spir_kernel void @foo(ptr addrspace(1) %p) {
  call void @llvm.prefetch.p1(ptr addrspace(1) %p, i32 0, i32 3, i32 1)
  ret void
}

; NO-EXT: %[[#PTR:]] = OpTypePointer CrossWorkgroup
; NO-EXT: %[[#VOID:]] = OpTypeVoid
; NO-EXT: %[[#FTY:]] = OpTypeFunction %[[#VOID]] %[[#PTR]]
; NO-EXT: %[[#I64:]] = OpTypeInt 64 0
; NO-EXT: %[[#ONE:]] = OpConstant %[[#I64]] 1
; NO-EXT: %[[#]] = OpFunction %[[#VOID]] None %[[#FTY]]
; NO-EXT: %[[#P:]] = OpFunctionParameter %[[#PTR]]
; NO-EXT: OpExtInst %[[#VOID]] %[[#EXTSET]] prefetch %[[#P]] %[[#ONE]]

; EXT-NOT: OpExtInst{{.*}}prefetch
