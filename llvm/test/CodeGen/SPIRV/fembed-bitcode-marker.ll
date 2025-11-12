; Expanding the bitcode marker works only for AMD at the moment.
; RUN: not llc -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o -
; RUN: llc -verify-machineinstrs -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}
;
; Verify that we lower the embedded bitcode

@llvm.embedded.module = private addrspace(1) constant [0 x i8] zeroinitializer, section ".llvmbc", align 1
@llvm.compiler.used = appending addrspace(1) global [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @llvm.embedded.module to ptr addrspace(4))], section "llvm.metadata"

; CHECK: OpName %[[#LLVM_EMBEDDED_MODULE:]] "llvm.embedded.module"
; CHECK: OpDecorate %[[#LLVM_EMBEDDED_MODULE]] Constant
; CHECK: %[[#UCHAR:]] = OpTypeInt 8 0
; CHECK: %[[#UINT:]] = OpTypeInt 32 0
; CHECK: %[[#ONE:]] = OpConstant %[[#UINT]] 1
; CHECK: %[[#UCHAR_ARR_1:]] = OpTypeArray %[[#UCHAR]] %[[#ONE]]
; CHECK: %[[#UCHAR_ARR_1_PTR:]] = OpTypePointer CrossWorkgroup %[[#UCHAR_ARR_1]]
; CHECK: %[[#CONST_UCHAR_ARR_1:]] = OpConstantNull %[[#UCHAR_ARR_1]]
; CHECK: %[[#LLVM_EMBEDDED_MODULE]] = OpVariable %[[#UCHAR_ARR_1_PTR]] CrossWorkgroup %[[#CONST_UCHAR_ARR_1]]

define spir_kernel void @foo() {
entry:
  ret void
}
