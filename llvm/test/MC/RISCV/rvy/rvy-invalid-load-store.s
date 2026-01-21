## For RVY, the loads and stores with the zero register as the base are reserved
## since this is an always-trapping encoding that may be reused in the future.
## 1) Check that the assembler rejects zero reg base for capmode but allows it in integer mode.
## 2) Check that the disassembler with +cap-mode decodes zero reg base as invalid instructions
# RUN: llvm-mc --triple=riscv32 -mattr=+experimental-y --riscv-no-aliases --show-encoding < %s \
# RUN:   | FileCheck --check-prefixes=CHECK-ASM,CHECK %s
# RUN: not llvm-mc --triple=riscv32 -mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding < %s 2>&1 \
# RUN:   | FileCheck --check-prefixes=ERR-RVY --implicit-check-not=error: %s
# RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-y -o %t.rv32i.o < %s
# RUN: llvm-objdump --mattr=+experimental-y -M no-aliases -d --no-print-imm-hex %t.rv32i.o \
# RUN:   | FileCheck --check-prefixes=CHECK %s
# RUN: llvm-objdump --mattr=+experimental-y,+cap-mode -M no-aliases -d --no-print-imm-hex %t.rv32i.o \
# RUN:   | FileCheck --check-prefixes=CHECK-CAP-MODE-DISASM %s

# RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y --riscv-no-aliases --show-encoding --defsym=RV64=1 < %s \
# RUN:   | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-64,CHECK,CHECK-64 %s
# RUN: not llvm-mc --triple=riscv64 --mattr=+experimental-y,+cap-mode --riscv-no-aliases --show-encoding --defsym=RV64=1 < %s 2>&1 \
# RUN:   | FileCheck --check-prefixes=ERR-RVY,ERR-RVY-64 --implicit-check-not=error: %s
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-y --riscv-no-aliases --show-encoding --defsym=RV64=1 -o %t.rv64i.o < %s
# RUN: llvm-objdump --mattr=+experimental-y -M no-aliases -d --no-print-imm-hex %t.rv64i.o \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-64 %s
# RUN: llvm-objdump --mattr=+experimental-y,+cap-mode -M no-aliases -d --no-print-imm-hex %t.rv64i.o \
# RUN:   | FileCheck --check-prefixes=CHECK-CAP-MODE-DISASM,CHECK-CAP-MODE-DISASM-64 %s

# CHECK: lb	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x03,0x05,0x00,0x01]
# CHECK-CAP-MODE-DISASM: 01000503 <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
lb a0, 16(zero)
# CHECK-NEXT: sb	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x23,0x08,0xa0,0x00]
# CHECK-CAP-MODE-DISASM-NEXT: 00a00823 <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
sb a0, 16(zero)
# CHECK-NEXT: lbu	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x03,0x45,0x00,0x01]
# CHECK-CAP-MODE-DISASM-NEXT: 01004503 <unknown>
# ERR-RVY: :[[#@LINE+1]]:12: error: register must be a GPR excluding zero (x0)
lbu a0, 16(zero)
# CHECK-NEXT: lh	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x03,0x15,0x00,0x01]
# CHECK-CAP-MODE-DISASM-NEXT: 01001503 <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
lh a0, 16(zero)
# CHECK-NEXT: sh	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x23,0x18,0xa0,0x00]
# CHECK-CAP-MODE-DISASM-NEXT: 00a01823 <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
sh a0, 16(zero)
# CHECK-NEXT: lhu	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x03,0x55,0x00,0x01]
# CHECK-CAP-MODE-DISASM-NEXT: 01005503 <unknown>
# ERR-RVY: :[[#@LINE+1]]:12: error: register must be a GPR excluding zero (x0)
lhu a0, 16(zero)
# CHECK-NEXT: lw	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x03,0x25,0x00,0x01]
# CHECK-CAP-MODE-DISASM-NEXT: 01002503 <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
lw a0, 16(zero)
# CHECK-NEXT: sw	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x23,0x28,0xa0,0x00]
# CHECK-CAP-MODE-DISASM-NEXT: 00a02823 <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
sw a0, 16(zero)
#
.ifdef RV64
# CHECK-64: lwu	a0, 16(zero)
# CHECK-ASM-64-SAME: # encoding: [0x03,0x65,0x00,0x01]
# CHECK-CAP-MODE-DISASM-64-NEXT: 01006503 <unknown>
# ERR-RVY-64: :[[#@LINE+1]]:12: error: register must be a GPR excluding zero (x0)
lwu a0, 16(zero)
# CHECK-64: ld	a0, 16(zero)
# CHECK-ASM-64-SAME: # encoding:  [0x03,0x35,0x00,0x01]
# CHECK-CAP-MODE-DISASM-64-NEXT: 01003503 <unknown>
# ERR-RVY-64: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
ld a0, 16(zero)
# CHECK-64: sd	a0, 16(zero)
# CHECK-ASM-64-SAME: # encoding: [0x23,0x38,0xa0,0x00]
# CHECK-CAP-MODE-DISASM-64-NEXT: 00a03823 <unknown>
# ERR-RVY-64: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
sd a0, 16(zero)
.endif
#
# CHECK-NEXT: ly	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x0f,0x45,0x00,0x01]
# CHECK-CAP-MODE-DISASM-NEXT: 0100450f <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
ly a0, 16(zero)
# CHECK-NEXT: sy	a0, 16(zero)
# CHECK-ASM-SAME: # encoding: [0x23,0x48,0xa0,0x00]
# CHECK-CAP-MODE-DISASM-NEXT: 00a04823 <unknown>
# ERR-RVY: :[[#@LINE+1]]:11: error: register must be a GPR excluding zero (x0)
sy a0, 16(zero)
