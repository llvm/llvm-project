; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: icmp requires integer operands
define void @byte_compare(b8 %b1, b8 %b2) {
  %cmp = icmp eq b8 %b1, %b2
  ret void
}
