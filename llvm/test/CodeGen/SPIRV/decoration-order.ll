; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; This test checks the OpDecorate MIR is generated after the associated
; vreg definition in the case of an array size declared through this lowering.

define spir_func i32 @foo() {
entry:
  %var = alloca i64
  br label %block

block:
  call void @llvm.memset.p0.i64(ptr align 8 %var, i8 0, i64 24, i1 false)
  ret i32 0
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
