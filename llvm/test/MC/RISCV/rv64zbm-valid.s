# With B extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip matix extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbm -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbm < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbm -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: bmatflip t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x33,0x60]
bmatflip t0, t1
# CHECK-ASM-AND-OBJ: bmator t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x32,0x73,0x08]
bmator t0, t1, t2
# CHECK-ASM-AND-OBJ: bmatxor t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x32,0x73,0x48]
bmatxor t0, t1, t2
