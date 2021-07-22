; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from 'i32' to 'b8'
define void @invalid_trunc(i32 %v) {
  %t = trunc i32 %v to b8
  ret void
}
