# RUN: not llvm-mca -march=aarch64 -mcpu=cortex-a57 -resource-pressure=false < %s 2> %t
# RUN: FileCheck --input-file %t %s

  add	x0, x1, x2, lsl #3

# CHECK: error
# CHECK-SAME: unable to resolve scheduling class for write variant.
