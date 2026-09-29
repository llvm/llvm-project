# Check parsing symbol expressions

# RUN: llvm-mc -triple=mipsel -show-inst-operands %s 2> %t0
# RUN: FileCheck %s < %t0

    .global __start
    .ent    __start
__start:
    nop
loc:
    jal     __start + 0x4       # CHECK: instruction: [jal, Imm<__start+4>]
    jal     __start + (-0x10)   # CHECK: instruction: [jal, Imm<__start-16>]
    jal     (__start + (-0x10)) # CHECK: instruction: [jal, Imm<__start-16>]
    .end    __start
