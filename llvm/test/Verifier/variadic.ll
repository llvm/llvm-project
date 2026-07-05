; RUN: not opt -S -passes=verify 2>&1 < %s | FileCheck %s

; CHECK: va_start called in a non-varargs function
declare void @llvm.va_start(ptr)
define void @not_vararg(ptr %p) nounwind {
  call void @llvm.va_start(ptr %p)
  ret void
}
