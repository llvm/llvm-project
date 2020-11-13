# With B extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip ternary extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbt -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbt < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbt -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: cmix t0, t1, t2, t3
# CHECK-ASM: encoding: [0xb3,0x92,0x63,0xe6]
cmix t0, t1, t2, t3
# CHECK-ASM-AND-OBJ: cmov t0, t1, t2, t3
# CHECK-ASM: encoding: [0xb3,0xd2,0x63,0xe6]
cmov t0, t1, t2, t3
# CHECK-ASM-AND-OBJ: fsl t0, t1, t2, t3
# CHECK-ASM: encoding: [0xb3,0x12,0xc3,0x3d]
fsl t0, t1, t2, t3
# CHECK-ASM-AND-OBJ: fsr t0, t1, t2, t3
# CHECK-ASM: encoding: [0xb3,0x52,0xc3,0x3d]
fsr t0, t1, t2, t3
# CHECK-ASM-AND-OBJ: fsri t0, t1, t2, 0
# CHECK-ASM: encoding: [0x93,0x52,0x03,0x3c]
fsri t0, t1, t2, 0
