# With B extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbb < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbb -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: addiwu t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x42,0x03,0x00]
addiwu t0, t1, 0
# CHECK-ASM-AND-OBJ: slliu.w t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x08]
slliu.w t0, t1, 0
# CHECK-ASM-AND-OBJ: addwu t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x0a]
addwu t0, t1, t2
# CHECK-ASM-AND-OBJ: subwu t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x4a]
subwu t0, t1, t2
# CHECK-ASM-AND-OBJ: addu.w t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x08]
addu.w t0, t1, t2
# CHECK-ASM-AND-OBJ: subu.w t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x02,0x73,0x48]
subu.w t0, t1, t2
# CHECK-ASM-AND-OBJ: slow t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x20]
slow t0, t1, t2
# CHECK-ASM-AND-OBJ: srow t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x52,0x73,0x20]
srow t0, t1, t2
# CHECK-ASM-AND-OBJ: sloiw t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x20]
sloiw t0, t1, 0
# CHECK-ASM-AND-OBJ: sroiw t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x52,0x03,0x20]
sroiw t0, t1, 0
# CHECK-ASM-AND-OBJ: clzw t0, t1
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x60]
clzw t0, t1
# CHECK-ASM-AND-OBJ: ctzw t0, t1
# CHECK-ASM: encoding: [0x9b,0x12,0x13,0x60]
ctzw t0, t1
# CHECK-ASM-AND-OBJ: pcntw t0, t1
# CHECK-ASM: encoding: [0x9b,0x12,0x23,0x60]
pcntw t0, t1
