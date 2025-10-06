; RUN: llc -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

@llvm.embedded.module = private constant [4 x i8] c"BC\C0\DE", section ".llvmbc", align 1
@llvm.cmdline = private constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
@llvm.compiler.used = appending global [2 x ptr] [ptr @llvm.embedded.module, ptr @llvm.cmdline], section "llvm.metadata"

; CHECK-DAG: OpName [[FOO:%[0-9]+]] "foo"
; CHECK-DAG: OpName [[MODULE:%[0-9]+]] "llvm.embedded.module"
; CHECK-DAG: [[INT8:%[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: [[INT32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[CONST_4_32:%[0-9]+]] = OpConstant [[INT32]] 4
; CHECK-DAG: [[ARRAY_INT8x4:%[0-9]+]] = OpTypeArray [[INT8]] [[CONST_4_32]]
; CHECK-DAG: [[POINTER:%[0-9]+]] = OpTypePointer Function [[ARRAY_INT8x4]]
; CHECK-DAG: [[CONST_B_8:%[0-9]+]] = OpConstant [[INT8]] 66
; CHECK-DAG: [[CONST_C_8:%[0-9]+]] = OpConstant [[INT8]] 67
; CHECK-DAG: [[CONST_0xC0_8:%[0-9]+]] = OpConstant [[INT8]] 192
; CHECK-DAG: [[CONST_0xDE_8:%[0-9]+]] = OpConstant [[INT8]] 222
; CHECK-DAG: [[EMBEDDED_MODULE_INIT:%[0-9]+]] = OpConstantComposite [[ARRAY_INT8x4]] [[CONST_B_8]] [[CONST_C_8]] [[CONST_0xC0_8]] [[CONST_0xDE_8]]
; CHECK: [[FOO]] = OpFunction {{.*}} None {{.*}} 
; CHECK-DAG: {{%[0-9]+}} = OpVariable [[POINTER]] Function [[EMBEDDED_MODULE_INIT]]

define spir_kernel void @foo() {
entry:
  ret void
}
