# RUN: llvm-mc %s -triple=riscv32 -mattr=+f,+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-ASM %s
# RUN: llvm-mc %s -triple riscv64 -mattr=+f,+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+f,+c < %s \
# RUN:     | llvm-objdump --mattr=+f,+c -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+f,+c < %s \
# RUN:     | llvm-objdump --mattr=+f,+c -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-OBJ %s

target:

# CHECK-ASM: .insn cr  2, 9, a0, a1
# CHECK-ASM: encoding: [0x2e,0x95]
# CHECK-OBJ: c.add a0, a1
.insn cr  2, 9, a0, a1

# CHECK-ASM: .insn cr  2, 9, a0, a1
# CHECK-ASM: encoding: [0x2e,0x95]
# CHECK-OBJ: c.add a0, a1
.insn cr C2, 9, a0, a1

# CHECK-ASM: .insn ci  1, 0, a0, 13
# CHECK-ASM: encoding: [0x35,0x05]
# CHECK-OBJ: c.addi a0, 0xd
.insn ci  1, 0, a0, 13

# CHECK-ASM: .insn ci  1, 0, a0, 13
# CHECK-ASM: encoding: [0x35,0x05]
# CHECK-OBJ: c.addi a0, 0xd
.insn ci C1, 0, a0, 13

# CHECK-ASM: .insn ciw  0, 0, a0, 13
# CHECK-ASM: encoding: [0xa8,0x01]
# CHECK-OBJ: c.addi4spn a0, sp, 0xc8
.insn ciw  0, 0, a0, 13

# CHECK-ASM: .insn ciw  0, 0, a0, 13
# CHECK-ASM: encoding: [0xa8,0x01]
# CHECK-OBJ: c.addi4spn a0, sp, 0xc8
.insn ciw C0, 0, a0, 13

# CHECK-ASM: .insn css  2, 6, a0, 13
# CHECK-ASM: encoding: [0xaa,0xc6]
# CHECK-OBJ: c.swsp a0, 0x4c(sp)
.insn css  2, 6, a0, 13

# CHECK-ASM: .insn cl  0, 2, a0, 13
# CHECK-ASM: encoding: [0xa8,0x4d]
# CHECK-OBJ: c.lw a0, 0x58(a1)
.insn cl  0, 2, a0, 13(a1)

# CHECK-ASM: .insn cl  0, 2, a0, 0
# CHECK-ASM: encoding: [0x88,0x41]
# CHECK-OBJ: c.lw a0, 0x0(a1)
.insn cl  0, 2, a0, 0(a1)

# CHECK-ASM: .insn cs  0, 6, a0, 13
# CHECK-ASM: encoding: [0xa8,0xcd]
# CHECK-OBJ: c.sw a0, 0x58(a1)
.insn cs  0, 6, a0, 13(a1)

# CHECK-ASM: .insn cs  0, 6, a0, 0
# CHECK-ASM: encoding: [0x88,0xc1]
# CHECK-OBJ: c.sw a0, 0x0(a1)
.insn cs  0, 6, a0, (a1)

# CHECK-ASM: .insn ca  1, 35, 0, a0, a1
# CHECK-ASM: encoding: [0x0d,0x8d]
# CHECK-OBJ: c.sub a0, a1
.insn ca  1, 35, 0, a0, a1

# CHECK-ASM: .insn cb 1, 6, a0, target
# CHECK-ASM: encoding: [0x01'A',0xc1'A']
# CHECK-OBJ: c.beqz a0, 0x0 <target>
.insn cb  1, 6, a0, target

# CHECK-ASM: .insn cj 1, 5, target
# CHECK-ASM: encoding: [0bAAAAAA01,0b101AAAAA]
# CHECK-OBJ: c.j 0x0 <target>
.insn cj  1, 5, target
