# With B extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip extract/deposit extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbe -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbe < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbe -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: bdep t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x62,0x73,0x48]
bdep t0, t1, t2
# CHECK-ASM-AND-OBJ: bext t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x62,0x73,0x08]
bext t0, t1, t2
