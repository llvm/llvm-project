# With Bit-Field extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbf -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbf < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbf -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: bfpw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x72,0x73,0x48]
bfpw t0, t1, t2
