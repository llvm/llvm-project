# With B extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip single bit extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbs -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbs < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbs -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: bclrw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x48]
bclrw t0, t1, t2
# CHECK-ASM-AND-OBJ: bsetw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x28]
bsetw t0, t1, t2
# CHECK-ASM-AND-OBJ: binvw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x68]
binvw t0, t1, t2
# CHECK-ASM-AND-OBJ: bextw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x52,0x73,0x48]
bextw t0, t1, t2
# CHECK-ASM-AND-OBJ: bclriw  t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x48]
bclriw	t0, t1, 0
# CHECK-ASM-AND-OBJ: bsetiw  t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x28]
bsetiw	t0, t1, 0
# CHECK-ASM-AND-OBJ: binviw  t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x68]
binviw	t0, t1, 0
