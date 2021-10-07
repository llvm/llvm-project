# With Bitmanip ternary extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbt -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbt < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbt -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fslw t0, t1, t2, t3
# CHECK-ASM: encoding: [0xbb,0x12,0xc3,0x3d]
fslw t0, t1, t2, t3
# CHECK-ASM-AND-OBJ: fsrw t0, t1, t2, t3
# CHECK-ASM: encoding: [0xbb,0x52,0xc3,0x3d]
fsrw t0, t1, t2, t3
# CHECK-ASM-AND-OBJ: fsriw t0, t1, t2, 0
# CHECK-ASM: encoding: [0x9b,0x52,0x03,0x3c]
fsriw t0, t1, t2, 0
