# RUN: not llvm-mca -march=aarch64 -mcpu=cyclone -resource-pressure=false < %s 2> %t
# RUN: FileCheck --input-file %t %s

  ldr	x7, [x1, #8]
  ldr	x6, [x1, x2]
  ldr	x4, [x1, x2, sxtx]

# CHECK: error
# CHECK-SAME: unable to resolve scheduling class for write variant.
