# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding -mattr=+v \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding -mattr=+v --print-imm-hex \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-HEX %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+v < %s \
# RUN:     | llvm-objdump -M no-aliases --mattr=+v --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+v < %s \
# RUN:     | llvm-objdump -M no-aliases --mattr=+v --print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-HEX %s

# CHECK-ASM: beq s1, s1, 102
# CHECK-ASM-HEX: beq s1, s1, 0x66
# CHECK-OBJ: beq s1, s1, 0x66
# CHECK-OBJ-HEX: beq s1, s1, 0x66
beq s1, s1, 102

_sym:
# CHECK-ASM: beq s1, s1, _sym
# CHECK-ASM-HEX: beq s1, s1, _sym
# CHECK-OBJ: beq s1, s1, 0x4
# CHECK-OBJ-HEX: beq s1, s1, 0x4
beq s1, s1, _sym

# CHECK-ASM: lw a0, 97(a2)
# CHECK-ASM-HEX: lw a0, 0x61(a2)
# CHECK-OBJ: lw a0, 97(a2)
# CHECK-OBJ-HEX: lw a0, 0x61(a2)
lw a0, 97(a2)

# CHECK-ASM: csrrwi t0, 4095, 31
# CHECK-ASM-HEX: csrrwi t0, 0xfff, 0x1f
# CHECK-OBJ: csrrwi t0, 4095, 31
# CHECK-OBJ-HEX: csrrwi t0, 0xfff, 0x1f
csrrwi t0, 0xfff, 31


# CHECK-ASM: vsetvli a2, a0, 255
# CHECK-ASM-HEX: vsetvli a2, a0, 0xff
# CHECK-OBJ: vsetvli a2, a0, 255
# CHECK-OBJ-HEX: vsetvli a2, a0, 0xff
vsetvli a2, a0, 0xff
