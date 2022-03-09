## Test valid barrier instructions.

# RUN: llvm-mc %s --triple=loongarch32 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc %s --triple=loongarch64 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s

# CHECK-ASM: dbar 0
# CHECK-ASM: encoding: [0x00,0x00,0x72,0x38]
dbar 0

# CHECK-ASM: ibar 0
# CHECK-ASM: encoding: [0x00,0x80,0x72,0x38]
ibar 0

