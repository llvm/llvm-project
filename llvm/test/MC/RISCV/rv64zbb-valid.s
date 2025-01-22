# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zbb -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zbb < %s \
# RUN:     | llvm-objdump --mattr=+zbb -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xbb,0x42,0x03,0x08]
zext.h t0, t1
# CHECK-ASM-AND-OBJ: rori t0, t1, 63
# CHECK-ASM: encoding: [0x93,0x52,0xf3,0x63]
rori t0, t1, 63
# CHECK-ASM-AND-OBJ: rev8 t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0x83,0x6b]
rev8 t0, t1

# CHECK-ASM-AND-OBJ: clzw t0, t1
# CHECK-ASM: encoding: [0x9b,0x12,0x03,0x60]
clzw t0, t1
# CHECK-ASM-AND-OBJ: ctzw t0, t1
# CHECK-ASM: encoding: [0x9b,0x12,0x13,0x60]
ctzw t0, t1
# CHECK-ASM-AND-OBJ: cpopw t0, t1
# CHECK-ASM: encoding: [0x9b,0x12,0x23,0x60]
cpopw t0, t1

# CHECK-ASM-AND-OBJ: addi t0, zero, -18
# CHECK-ASM-AND-OBJ: rori t0, t0, 21
li t0, -149533581377537
# CHECK-ASM-AND-OBJ: addi t0, zero, -86
# CHECK-ASM-AND-OBJ: rori t0, t0, 4
li t0, -5764607523034234886
# CHECK-ASM-AND-OBJ: addi t0, zero, -18
# CHECK-ASM-AND-OBJ: rori t0, t0, 37
li t0, -2281701377

# CHECK-ASM-AND-OBJ: rolw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x60]
rolw t0, t1, t2
# CHECK-ASM-AND-OBJ: rorw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x52,0x73,0x60]
rorw t0, t1, t2
# CHECK-ASM-AND-OBJ: roriw t0, t1, 31
# CHECK-ASM: encoding: [0x9b,0x52,0xf3,0x61]
roriw t0, t1, 31
# CHECK-ASM-AND-OBJ: roriw t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x52,0x03,0x60]
roriw t0, t1, 0
