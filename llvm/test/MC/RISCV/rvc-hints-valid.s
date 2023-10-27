# RUN: llvm-mc %s -triple=riscv32 -mattr=+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple riscv64 -mattr=+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: c.nop 8
# CHECK-ASM: encoding: [0x21,0x00]
c.nop 8

# CHECK-ASM-AND-OBJ: c.nop 7
# CHECK-ASM: encoding: [0x1d,0x00]
c.addi x0, 7

# CHECK-ASM-AND-OBJ: c.addi a0, 0
# CHECK-ASM: encoding: [0x01,0x05]
c.addi a0, 0

# CHECK-ASM-AND-OBJ: c.li zero, 0
# CHECK-ASM: encoding: [0x01,0x40]
c.li x0, 0

# CHECK-ASM-AND-OBJ: c.li zero, 1
# CHECK-ASM: encoding: [0x05,0x40]
c.li x0, 1

# CHECK-ASM-AND-OBJ: c.lui zero, 1
# CHECK-ASM: encoding: [0x05,0x60]
c.lui x0, 1

# CHECK-ASM-AND-OBJ: c.mv zero, a0
# CHECK-ASM: encoding: [0x2a,0x80]
c.mv x0, a0

# CHECK-ASM-AND-OBJ: c.add zero, a0
# CHECK-ASM: encoding: [0x2a,0x90]
c.add x0, a0

# CHECK-ASM-AND-OBJ: c.slli zero, 1
# CHECK-ASM: encoding: [0x06,0x00]
c.slli x0, 1

# CHECK-ASM-AND-OBJ: c.slli64 zero
# CHECK-ASM: encoding: [0x02,0x00]
c.slli64 x0

# CHECK-ASM-AND-OBJ: c.slli64 a0
# CHECK-ASM: encoding: [0x02,0x05]
c.slli64 a0

# CHECK-ASM-AND-OBJ: c.srli64 a1
# CHECK-ASM: encoding: [0x81,0x81]
c.srli64 a1

# CHECK-ASM-AND-OBJ: c.srai64 a0
# CHECK-ASM: encoding: [0x01,0x85]
c.srai64 a0
