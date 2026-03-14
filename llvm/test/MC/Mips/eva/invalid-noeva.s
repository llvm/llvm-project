# invalid operand for instructions that are invalid without -mattr=+eva flag
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r3 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r5 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r6 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r2 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r3 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r5 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r6 2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        tlbinv                         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        tlbinvf                        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
