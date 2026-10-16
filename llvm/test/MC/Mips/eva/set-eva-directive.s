# invalid operand for instructions that are invalid without -mattr=+eva flag
#
# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r6 | FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r6 | FileCheck %s

        .set noat
        .set eva
        tlbinv                         # CHECK: tlbinv                  # encoding: [0x42,0x00,0x00,0x03]
        tlbinvf                        # CHECK: tlbinvf                 # encoding: [0x42,0x00,0x00,0x04]
