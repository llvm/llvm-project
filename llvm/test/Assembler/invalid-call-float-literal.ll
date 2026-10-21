; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: unexpected floating-point literal

define void @foo() {
  call void f0x12345678()
}
