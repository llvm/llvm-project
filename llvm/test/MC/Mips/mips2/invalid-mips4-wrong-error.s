# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        bc1fl     $fcc7,27        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        bc1tl     $fcc7,27        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ld        $sp,-28645($s1) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwu       $s3,-24086($v1) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        scd       $15,-8243($sp)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        sd        $12,5835($10)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdl       $a3,-20961($s8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdr       $11,-20423($12) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
