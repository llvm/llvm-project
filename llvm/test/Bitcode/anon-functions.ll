; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; PR25101

; CHECK: define void @0()
define void @0() {
  ret void
}

; CHECK: define void @f()
define void @f() {
  ret void
}

; CHECK: define void @1()
define void @1() {
  ret void
}

