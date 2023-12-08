; Check upgrade is removing the incompatible attributes on void return type.

; RUN: llvm-dis < %s.bc | FileCheck %s

; CHECK: define void @f()
define align 8 void @f() {
  ret void
}

define void @g() {
; CHECK: call void @f()
  call align 8 void @f();
  ret void
}
