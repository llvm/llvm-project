# Instructions that are invalid but currently emit the wrong error message.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        bc1any2f  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc1any2t  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc1any4f  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc1any4t  $fcc2,4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        ldc2 $1, -2049($12)           # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        ldc2 $1, 2048($12)            # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        ldc2 $1, 1023($32)            # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        lwc2 $1, -2049($4)            # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        lwc2 $1, 2048($4)             # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        lwc2 $1, 16($32)              # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        sdc2 $1, -2049($16)           # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        sdc2 $1, 2048($16)            # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        sdc2 $1, 8($32)               # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
        swc2 $1, -2049($17)           # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        swc2 $1, 2048($17)            # CHECK: :[[@LINE]]:9: error: instruction requires a CPU feature not currently enabled
        swc2 $1, 777($32)             # CHECK: :[[@LINE]]:18: error: expected memory with 16-bit signed offset
