; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from '<4 x i64>' to 'i8'

define i8 @foo(<4 x i64> %x) {
  %y = trunc <4 x i64> %x to i8
  ret i8 %y
}
