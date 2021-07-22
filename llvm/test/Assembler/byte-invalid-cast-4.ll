; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from 'i8' to 'b32'
define void @invalid_zext(i8 %v) {
  %t = zext i8 %v to b32
  ret void
}
