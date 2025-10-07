; XFAIL: *
; RUN: llc -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@llvm.embedded.module = private constant [0 x i8] zeroinitializer, section ".llvmbc", align 1
@llvm.cmdline = private constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
@llvm.compiler.used = appending global [2 x ptr] [ptr @llvm.embedded.module, ptr @llvm.cmdline], section "llvm.metadata"

define spir_kernel void @foo() {
entry:
  ret void
}
