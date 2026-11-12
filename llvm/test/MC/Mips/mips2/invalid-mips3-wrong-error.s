# Instructions that are invalid and are correctly rejected but used to emit
# the wrong error message.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        dmult     $s7,$a5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldl       $t8,-4167($t8)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        ldr       $t2,-30358($s4)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        scd       $t3,-8243($sp)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sdl       $a3,-20961($s8)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sdr       $a7,-20423($t0)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
