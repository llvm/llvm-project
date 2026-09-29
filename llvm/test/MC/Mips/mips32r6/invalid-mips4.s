# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        bc1fl     $fcc7,27            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        bc1tl     $fcc7,27            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ldxc1     $f8,$s7($15)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lwxc1     $f12,$s1($s8)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdxc1     $f11,$10($14)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        swxc1     $f19,$12($k0)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        prefx     0,$2($31)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
