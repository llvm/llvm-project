; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from '<4 x ptr>' to '<2 x ptr>'
define <2 x ptr> @illegal_vector_pointer_bitcast_num_elements(<4 x ptr> %c) {
  %bc = bitcast <4 x ptr> %c to <2 x ptr>
  ret <2 x ptr> %bc
}
