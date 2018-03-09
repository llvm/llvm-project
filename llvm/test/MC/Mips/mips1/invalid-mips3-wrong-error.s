# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips1 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        ldc2      $8,-21181($at)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldc2      $20,-1024($s2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldl       $24,-4167($24)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldr       $14,-30358($s4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ll        $v0,-7321($s2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sc        $15,18904($s3)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        scd       $15,-8243($sp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdc2      $20,23157($s2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdc2      $20,-1024($s2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdl       $a3,-20961($s8)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdr       $11,-20423($12)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
