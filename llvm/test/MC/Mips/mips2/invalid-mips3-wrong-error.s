# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        dmult     $s7,$a5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ld        $sp,-28645($s1)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldl       $t8,-4167($t8)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldr       $t2,-30358($s4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lld       $zero,-14736($ra) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        lwu       $s3,-24086($v1)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        scd       $t3,-8243($sp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
        sd        $t0,5835($a6)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdl       $a3,-20961($s8)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdr       $a7,-20423($t0)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
