# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        bc2f  $fcc0,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2f  4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2fl $fcc1,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2fl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2t  $fcc0,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2t  4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2tl $fcc1,4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2tl 4                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
