# With B extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zba -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zba < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zba -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: slli.uw t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x08]
slli.uw t0, t1, 0
# CHECK-ASM-AND-OBJ: add.uw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x08]
add.uw t0, t1, t2
# CHECK-ASM-AND-OBJ: sh1add.uw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x22,0x73,0x20]
sh1add.uw t0, t1, t2
# CHECK-ASM-AND-OBJ: sh2add.uw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x42,0x73,0x20]
sh2add.uw t0, t1, t2
# CHECK-ASM-AND-OBJ: sh3add.uw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x62,0x73,0x20]
sh3add.uw t0, t1, t2
