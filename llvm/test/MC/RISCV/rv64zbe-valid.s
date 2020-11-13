# With B extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip extract/deposit extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbe -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbe < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbe -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: bdepw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x62,0x73,0x48]
bdepw t0, t1, t2
# CHECK-ASM-AND-OBJ: bextw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x62,0x73,0x08]
bextw t0, t1, t2
