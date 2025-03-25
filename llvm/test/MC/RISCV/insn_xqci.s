# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilia,+experimental-xqcilo,+experimental-xqcibi,+experimental-xqcilb \
# RUN:         -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s

# CHECK-ASM: .insn qc.eai 31, 2, 0, a0, 16711935
# CHECK-ASM: encoding: [0x1f,0x25,0xff,0x00,0xff,0x00]
# CHECK-OBJ: qc.e.addai a0, 0xff00ff
.insn qc.eai 0x1f, 2, 0, a0, 0x00FF00FF

# CHECK-ASM: .insn qc.ei 31, 3, 2, a0, a1, 16711935
# CHECK-ASM: encoding: [0x1f,0xb5,0xf5,0x8f,0xc0,0x3f]
# CHECK-OBJ: qc.e.addi a0, a1, 0xff00ff
.insn qc.ei 0x1f, 3, 2, a0, a1, 0x00FF00FF

# CHECK-ASM: .insn qc.ei 31, 5, 0, a1, 16711935(a0)
# CHECK-ASM: encoding: [0x9f,0x55,0xf5,0x0f,0xc0,0x3f]
# CHECK-OBJ: qc.e.lb a1, 0xff00ff(a0)
.insn qc.ei 0x1f, 5, 0, a1, 0x00FF00FF(a0)

# CHECK-ASM: .insn qc.ei 31, 5, 0, a1, 0(a0)
# CHECK-ASM: encoding: [0x9f,0x55,0x05,0x00,0x00,0x00]
# CHECK-OBJ: qc.e.lb a1, 0x0(a0)
.insn qc.ei 0x1f, 5, 0, a1, (a0)

# CHECK-ASM: .insn qc.eb 31, 4, 24, a0, 17476, 22
# CHECK-ASM: encoding: [0x1f,0x4b,0x85,0x01,0x44,0x44]
# CHECK-OBJ: qc.e.beqi a0, 0x4444, 0x2e
.insn qc.eb 0x1f, 4, 24, a0, 0x4444, 22

# CHECK-ASM: .insn qc.ej 31, 4, 0, 0, 22
# CHECK-ASM: encoding: [0x1f,0x4b,0x00,0x00,0x00,0x00]
# CHECK-OBJ: qc.e.j 0x34
.insn qc.ej 0x1f, 4, 0, 0, 22

# CHECK-ASM: .insn qc.es 31, 6, 1, a1, 0(a0)
# CHECK-ASM: encoding: [0x1f,0x60,0xb5,0x40,0x00,0x00]
# CHECK-OBJ: qc.e.sb a1, 0x0(a0)
.insn qc.es 0x1f, 6, 1, a1, (a0)
