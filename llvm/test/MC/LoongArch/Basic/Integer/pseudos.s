## Test valid pseudo instructions

# RUN: llvm-mc %s --triple=loongarch32 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc %s --triple=loongarch64 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s

# CHECK-ASM: nop
# CHECK-ASM: encoding: [0x00,0x00,0x40,0x03]
nop

# CHECK-ASM: move $a4, $a5
# CHECK-ASM: encoding: [0x28,0x01,0x15,0x00]
move $a4, $a5
