; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: invalid operand type for instruction
define void @invalid_lshr(b32 %b) {
  %t = lshr b32 %b, 8
  ret void
}
