# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        bc2f      4                   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        bc2t      4                   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: unknown instruction
        lwl       $s4,-4231($15)      # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwr       $zero,-19147($gp)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swl       $15,13694($s3)      # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swr       $s1,-26590($14)     # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        lwle      $s4,-4231($15)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwre      $zero,-19147($gp)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swle      $15,13694($s3)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        swre      $24, 5($3)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swre      $s1,-26590($14)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        lwc3      $12, 4($4)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
        swc3      $12, 4($4)          # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled
