# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -show-encoding  < %s 2>%t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s &>%t
# RUN: FileCheck --check-prefix=CHECK-MVE < %t %s

# CHECK-MVE: instruction requires: mve.fp
# CHECK: invalid instruction
vcadd.f32 q1, q2, q3, #270

# CHECK-MVE: instruction requires: mve.fp
# CHECK: invalid instruction
vadd.f32 q1, q2, q3

# CHECK-MVE: vadd.i16 q1, q2, q3 @ encoding: [0x14,0xef,0x46,0x28]
# CHECK: invalid instruction
vadd.i16 q1, q2, q3
