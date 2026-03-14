; RUN: llc -mtriple arm64-windows %s -o - 2>&1 | FileCheck %s

define ptr @foo(ptr %a) {
; CHECK-LABEL: foo
; CHECK-NOT: llvm.x86.seh.recoverfp
  %1 = call ptr @llvm.x86.seh.recoverfp(ptr @f, ptr %a)
  ret ptr %1
}

declare ptr @llvm.x86.seh.recoverfp(ptr, ptr)
declare i32 @f()
