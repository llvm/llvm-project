// RUN: not llvm-mc -o - -triple arm -mattr=+v8.6a -show-encoding %s 2>&1 | FileCheck %s
vfmat.bf16 d0, d0, d0
vfmat.bf16 d0, d0, q0
vfmat.bf16 d0, q0, d0
vfmat.bf16 q0, d0, d0
vfmat.bf16 q0, q0, d0
vfmat.bf16 q0, d0, q0
vfmat.bf16 d0, q0, q0
vfmat.bf16 q0, q0, q0[3]
vfmat.bf16 q0, q0, q0[3]
vfmat.bf16 q0, d0, d0[0]
vfmat.bf16 d0, q0, d0[0]
vfmat.bf16 q0, d0, d0[9]

vfmab.bf16 d0, d0, d0
vfmab.bf16 d0, d0, q0
vfmab.bf16 d0, q0, d0
vfmab.bf16 q0, d0, d0
vfmab.bf16 q0, q0, d0
vfmab.bf16 q0, d0, q0
vfmab.bf16 d0, q0, q0
vfmab.bf16 q0, q0, q0[3]
vfmab.bf16 q0, q0, q0[3]
vfmab.bf16 q0, d0, d0[0]
vfmab.bf16 d0, q0, d0[0]
vfmab.bf16 q0, d0, d0[9]

//CHECK:error: invalid instruction
//CHECK-NEXT:vfmat.bf16 d0, d0, d0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmat.bf16 d0, d0, q0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmat.bf16 d0, q0, d0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmat.bf16 q0, d0, d0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction, any one of the following would fix this:
//CHECK-NEXT:vfmat.bf16 q0, q0, d0
//CHECK-NEXT:^
//CHECK-NEXT:note: too few operands for instruction
//CHECK-NEXT:vfmat.bf16 q0, q0, d0
//CHECK-NEXT:                      ^
//CHECK-NEXT:note: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmat.bf16 q0, q0, d0
//CHECK-NEXT:                    ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmat.bf16 q0, d0, q0
//CHECK-NEXT:                ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmat.bf16 d0, q0, q0
//CHECK-NEXT:            ^
//CHECK-NEXT:error: invalid instruction, any one of the following would fix this:
//CHECK-NEXT:vfmat.bf16 q0, q0, q0[3]
//CHECK-NEXT:^
//CHECK-NEXT:note: operand must be a register in range [d0, d7]
//CHECK-NEXT:vfmat.bf16 q0, q0, q0[3]
//CHECK-NEXT:                    ^
//CHECK-NEXT:note: too many operands for instruction
//CHECK-NEXT:vfmat.bf16 q0, q0, q0[3]
//CHECK-NEXT:                      ^
//CHECK-NEXT:error: invalid instruction, any one of the following would fix this:
//CHECK-NEXT:vfmat.bf16 q0, q0, q0[3]
//CHECK-NEXT:^
//CHECK-NEXT:note: operand must be a register in range [d0, d7]
//CHECK-NEXT:vfmat.bf16 q0, q0, q0[3]
//CHECK-NEXT:                    ^
//CHECK-NEXT:note: too many operands for instruction
//CHECK-NEXT:vfmat.bf16 q0, q0, q0[3]
//CHECK-NEXT:                      ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmat.bf16 q0, d0, d0[0]
//CHECK-NEXT:                ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmat.bf16 d0, q0, d0[0]
//CHECK-NEXT:            ^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmat.bf16 q0, d0, d0[9]
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmab.bf16 d0, d0, d0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmab.bf16 d0, d0, q0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmab.bf16 d0, q0, d0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmab.bf16 q0, d0, d0
//CHECK-NEXT:^
//CHECK-NEXT:error: invalid instruction, any one of the following would fix this:
//CHECK-NEXT:vfmab.bf16 q0, q0, d0
//CHECK-NEXT:^
//CHECK-NEXT:note: too few operands for instruction
//CHECK-NEXT:vfmab.bf16 q0, q0, d0
//CHECK-NEXT:                      ^
//CHECK-NEXT:note: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmab.bf16 q0, q0, d0
//CHECK-NEXT:                    ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmab.bf16 q0, d0, q0
//CHECK-NEXT:                ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmab.bf16 d0, q0, q0
//CHECK-NEXT:            ^
//CHECK-NEXT:error: invalid instruction, any one of the following would fix this:
//CHECK-NEXT:vfmab.bf16 q0, q0, q0[3]
//CHECK-NEXT:^
//CHECK-NEXT:note: operand must be a register in range [d0, d7]
//CHECK-NEXT:vfmab.bf16 q0, q0, q0[3]
//CHECK-NEXT:                    ^
//CHECK-NEXT:note: too many operands for instruction
//CHECK-NEXT:vfmab.bf16 q0, q0, q0[3]
//CHECK-NEXT:                      ^
//CHECK-NEXT:error: invalid instruction, any one of the following would fix this:
//CHECK-NEXT:vfmab.bf16 q0, q0, q0[3]
//CHECK-NEXT:^
//CHECK-NEXT:note: operand must be a register in range [d0, d7]
//CHECK-NEXT:vfmab.bf16 q0, q0, q0[3]
//CHECK-NEXT:                    ^
//CHECK-NEXT:note: too many operands for instruction
//CHECK-NEXT:vfmab.bf16 q0, q0, q0[3]
//CHECK-NEXT:                      ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmab.bf16 q0, d0, d0[0]
//CHECK-NEXT:                ^
//CHECK-NEXT:error: operand must be a register in range [q0, q15]
//CHECK-NEXT:vfmab.bf16 d0, q0, d0[0]
//CHECK-NEXT:            ^
//CHECK-NEXT:error: invalid instruction
//CHECK-NEXT:vfmab.bf16 q0, d0, d0[9]
