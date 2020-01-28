# RUN: not llvm-mc -triple powerpc64-unknown-unknown < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple powerpc64le-unknown-unknown < %s 2> %t
# RUN: FileCheck < %t %s

 # CHECK: error: invalid operand for instruction
paddi 1, 1, 32, 1

# CHECK: error: invalid operand for instruction
pld 1, 32(1), 1

