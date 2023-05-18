# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=octeon 2>%t1
# RUN: FileCheck %s < %t1

  .set  noat
  lwc3  $4, 0($5)  # CHECK: :{{[0-9]+}}:{{[0-9]+}}: error: invalid operand for instruction
  swc3  $4, 0($5)  # CHECK: :{{[0-9]+}}:{{[0-9]+}}: error: invalid operand for instruction
  ldc3  $4, 0($5)  # CHECK: :{{[0-9]+}}:{{[0-9]+}}: error: invalid operand for instruction
  sdc3  $4, 0($5)  # CHECK: :{{[0-9]+}}:{{[0-9]+}}: error: invalid operand for instruction
