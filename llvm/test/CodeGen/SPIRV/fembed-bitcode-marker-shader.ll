; RUN: llc -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@llvm.embedded.module = private constant [0 x i8] zeroinitializer, section ".llvmbc", align 1
@llvm.cmdline = private constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
@llvm.compiler.used = appending global [2 x ptr] [ptr @llvm.embedded.module, ptr @llvm.cmdline], section "llvm.metadata"

; CHECK-DAG: OpName [[FOO:%[0-9]+]] "foo"
; CHECK-DAG: OpName [[MODULE:%[0-9]+]] "llvm.embedded.module"
; CHECK-DAG: [[INT8:%[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: [[RUNTIME_ARRAY_INT8x1:%[0-9]+]] = OpTypeRuntimeArray [[INT8]]
; CHECK-DAG: [[POINTER:%[0-9]+]] = OpTypePointer Function [[RUNTIME_ARRAY_INT8x1]]
; CHECK-DAG: [[EMBEDDED_MODULE_INIT:%[0-9]+]] = OpConstantNull [[RUNTIME_ARRAY_INT8x1]]
; CHECK: [[FOO]] = OpFunction {{.*}} None {{.*}} 
; CHECK-DAG: {{%[0-9]+}} = OpVariable [[POINTER]] Function [[EMBEDDED_MODULE_INIT]]

define void @foo() #1 {
entry:
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
