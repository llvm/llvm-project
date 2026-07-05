# Instructions that are invalid and are correctly rejected but used to emit
# the wrong error message.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        bc1fl     $fcc7,27        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        bc1tl     $fcc7,27        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: non-zero fcc register doesn't exist in current ISA level
        scd       $15,-8243($sp)  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sdl       $a3,-20961($s8) # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        sdr       $11,-20423($12) # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
