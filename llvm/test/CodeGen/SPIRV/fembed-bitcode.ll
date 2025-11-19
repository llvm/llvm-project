; RUN: llc -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}
;
; Verify that we can lower the embedded module and cmdline.

@llvm.embedded.module = private addrspace(1) constant [4 x i8] c"BC\C0\DE", section ".llvmbc", align 1
@llvm.cmdline = private addrspace(1) constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
@llvm.compiler.used = appending addrspace(1) global [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @llvm.embedded.module to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @llvm.cmdline to ptr addrspace(4))], section "llvm.metadata"

; CHECK: OpName %[[#LLVM_EMBEDDED_MODULE:]] "llvm.embedded.module"
; CHECK: OpName %[[#LLVM_CMDLINE:]] "llvm.cmdline"
; CHECK: OpDecorate %[[#LLVM_EMBEDDED_MODULE]] Constant
; CHECK: OpDecorate %[[#LLVM_CMDLINE]] Constant
; CHECK: %[[#UCHAR:]] = OpTypeInt 8 0
; CHECK: %[[#UINT:]] = OpTypeInt 32 0
; CHECK: %[[#FIVE:]] = OpConstant %[[#UINT]] 5
; CHECK: %[[#UCHAR_ARR_5:]] = OpTypeArray %[[#UCHAR]] %[[#FIVE]]
; CHECK: %[[#FOUR:]] = OpConstant %[[#UINT]] 4
; CHECK: %[[#UCHAR_ARR_4:]] = OpTypeArray %[[#UCHAR]] %[[#FOUR]]
; CHECK: %[[#UCHAR_ARR_5_PTR:]] = OpTypePointer CrossWorkgroup %[[#UCHAR_ARR_5]]
; CHECK: %[[#UCHAR_ARR_4_PTR:]] = OpTypePointer CrossWorkgroup %[[#UCHAR_ARR_4]]
; CHECK: %[[#CONST_UCHAR_ARR_4:]] = OpConstantComposite %[[#UCHAR_ARR_4]]
; CHECK: %[[#LLVM_EMBEDDED_MODULE]] = OpVariable %[[#UCHAR_ARR_4_PTR]] CrossWorkgroup %[[#CONST_UCHAR_ARR_4]]
; CHECK: %[[#CONST_UCHAR_ARR_5:]] = OpConstantComposite %[[#UCHAR_ARR_5]]
; CHECK: %[[#LLVM_CMDLINE]] = OpVariable %[[#UCHAR_ARR_5_PTR]] CrossWorkgroup %[[#CONST_UCHAR_ARR_5]]

define spir_kernel void @foo() {
entry:
  ret void
}
