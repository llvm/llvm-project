; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; llvm.experimental.stackmap is a varg intrisic.
declare void @llvm.experimental.stackmap(i64, i32)

; llvm.donothing is *not* a vararg intrinsic.
declare void @llvm.donothing(...)

define void @foo1() {
  call void @llvm.experimental.stackmap(i64 0, i32 12)
; CHECK: intrinsic was not defined with variable arguments!
  ret void
}

define void @foo2() {
  call void (...) @llvm.donothing(i64 0, i64 1)
; CHECK: intrinsic was defined with variable arguments!
  ret void
}
