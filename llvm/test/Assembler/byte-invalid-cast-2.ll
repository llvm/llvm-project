; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from 'b8' to 'i32'
define void @invalid_sext(b8 %b) {
  %t = sext b8 %b to i32
  ret void
}
