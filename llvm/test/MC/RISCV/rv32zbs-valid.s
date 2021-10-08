# With Bitmanip single bit extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbs -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbs -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbs < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbs -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbs < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbs -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: bclr t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x48]
bclr t0, t1, t2
# CHECK-ASM-AND-OBJ: bset t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x28]
bset t0, t1, t2
# CHECK-ASM-AND-OBJ: binv t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x68]
binv t0, t1, t2
# CHECK-ASM-AND-OBJ: bext t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x48]
bext t0, t1, t2
# CHECK-ASM-AND-OBJ: bclri t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x48]
bclri t0, t1, 1
# CHECK-ASM-AND-OBJ: bseti t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x28]
bseti t0, t1, 1
# CHECK-ASM-AND-OBJ: binvi t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x68]
binvi t0, t1, 1
# CHECK-ASM-AND-OBJ: bexti t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x52,0x13,0x48]
bexti t0, t1, 1
