; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from 'b32' to 'i8'
define void @invalid_trunc(b32 %b) {
  %t = trunc b32 %b to i8
  ret void
}
