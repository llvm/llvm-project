# With B extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip permutation extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbp -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbp < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbp -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: gorc t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x28]
gorc t0, t1, t2
# CHECK-ASM-AND-OBJ: grev t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x68]
grev t0, t1, t2
# CHECK-ASM-AND-OBJ: gorci t0, t1, 0
# CHECK-ASM: encoding: [0x93,0x52,0x03,0x28]
gorci t0, t1, 0
# CHECK-ASM-AND-OBJ: grevi t0, t1, 0
# CHECK-ASM: encoding: [0x93,0x52,0x03,0x68]
grevi t0, t1, 0
# CHECK-ASM-AND-OBJ: shfl t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x08]
shfl t0, t1, t2
# CHECK-ASM-AND-OBJ: unshfl t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x08]
unshfl t0, t1, t2
# CHECK-ASM-AND-OBJ: shfli t0, t1, 0
# CHECK-ASM: encoding: [0x93,0x12,0x03,0x08]
shfli t0, t1, 0
# CHECK-ASM-AND-OBJ: unshfli t0, t1, 0
# CHECK-ASM: encoding: [0x93,0x52,0x03,0x08]
unshfli t0, t1, 0
