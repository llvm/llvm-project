# Instructions that are invalid and are correctly rejected but use the wrong
# error message at the moment.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips1 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        ldc2      $8,-21181($at)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldc2      $8,-1024($at)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ldc3      $29,-28645($s1) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        ll        $v0,-7321($s2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sc        $t7,18904($s3)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdc2      $20,23157($s2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdc2      $20,-1024($s2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
        sdc3      $12,5835($t2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
