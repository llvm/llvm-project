# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips5 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

        .set noat
        di      $s8                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        di                          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ei      $t6                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ei                          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ext     $1, $2, 4, 5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ins     $1, $2, 4, 5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
