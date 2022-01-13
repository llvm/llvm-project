# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zbb -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zbb < %s \
# RUN:     | llvm-objdump --mattr=+zbb -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

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
