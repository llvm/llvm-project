# With B extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip single bit extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbs -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbs < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbs -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sbclr t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x48]
sbclr t0, t1, t2
# CHECK-ASM-AND-OBJ: sbset t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x28]
sbset t0, t1, t2
# CHECK-ASM-AND-OBJ: sbinv t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x68]
sbinv t0, t1, t2
# CHECK-ASM-AND-OBJ: sbext t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x48]
sbext t0, t1, t2
# CHECK-ASM-AND-OBJ: sbclri t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x48]
sbclri t0, t1, 1
# CHECK-ASM-AND-OBJ: sbseti t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x28]
sbseti t0, t1, 1
# CHECK-ASM-AND-OBJ: sbinvi t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x68]
sbinvi t0, t1, 1
# CHECK-ASM-AND-OBJ: sbexti t0, t1, 1
# CHECK-ASM: encoding: [0x93,0x52,0x13,0x48]
sbexti t0, t1, 1
