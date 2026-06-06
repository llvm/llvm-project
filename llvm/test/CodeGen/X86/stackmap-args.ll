; RUN: not llc -mtriple=x86_64-apple-darwin -mcpu=corei7 < %s 2>&1 | FileCheck %s
; Tests error when we pass non-immediate parameters to @llvm.experiment.stackmap

define void @first_arg() {
; CHECK: immarg operand has non-immediate parameter
entry:
  ; First operand should be immediate
  %id = add i64 0, 0
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 %id, i32 0)
  ret void
}

define void @second_arg() {
; CHECK: immarg operand has non-immediate parameter
entry:
  ; Second operand should be immediate
  %numShadowByte = add i32 0, 0
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 %numShadowByte)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)