/// For RVY, the loads and stores with the zero register as the base are reserved
/// since this is an always-trapping encoding that may be reused in the future:
/// - Check that the assembler rejects zero reg base for capmode but allows it in integral pointer mode.
/// - Check that the disassembler with +experimental-y decodes zero reg base as invalid instructions.
// RUN: llvm-mc --triple=riscv32 --mattr=+experimental-zyhybrid --show-encoding < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM,CHECK %s
// RUN: llvm-mc --triple=riscv32 --mattr=+experimental-y --verify %s
// RUN: llvm-mc --filetype=obj --triple=riscv32 --mattr=+experimental-zyhybrid -o %t.rv32i.o < %s
// RUN: llvm-objdump -M no-aliases --mattr=+experimental-zyhybrid -d --no-print-imm-hex %t.rv32i.o \
// RUN:   | FileCheck --check-prefixes=CHECK %s
// RUN: llvm-objdump -M no-aliases --mattr=+experimental-y -d --no-print-imm-hex %t.rv32i.o \
// RUN:   | FileCheck --check-prefixes=CHECK-CAP-MODE-DISASM %s

// RUN: llvm-mc --triple=riscv64 --mattr=+experimental-zyhybrid --show-encoding --defsym=RV64=1 < %s \
// RUN:   | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-64,CHECK,CHECK-64 %s
// RUN: llvm-mc --triple=riscv64 --mattr=+experimental-y --defsym=RV64=1 --verify=expected,rv64 %s
// RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+experimental-zyhybrid --defsym=RV64=1 -o %t.rv64i.o < %s
// RUN: llvm-objdump -M no-aliases --mattr=+experimental-zyhybrid -d --no-print-imm-hex %t.rv64i.o \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-64 %s
// RUN: llvm-objdump -M no-aliases --mattr=+experimental-y -d --no-print-imm-hex %t.rv64i.o \
// RUN:   | FileCheck --check-prefixes=CHECK-CAP-MODE-DISASM,CHECK-CAP-MODE-DISASM-64 %s

// CHECK: lb	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x03,0x05,0x00,0x01]
// CHECK-CAP-MODE-DISASM: 01000503 <unknown>
lb a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: sb	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x23,0x08,0xa0,0x00]
// CHECK-CAP-MODE-DISASM-NEXT: 00a00823 <unknown>
sb a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: lbu	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x03,0x45,0x00,0x01]
// CHECK-CAP-MODE-DISASM-NEXT: 01004503 <unknown>
lbu a0, 16(zero) // expected-error:12 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: lh	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x03,0x15,0x00,0x01]
// CHECK-CAP-MODE-DISASM-NEXT: 01001503 <unknown>
lh a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: sh	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x23,0x18,0xa0,0x00]
// CHECK-CAP-MODE-DISASM-NEXT: 00a01823 <unknown>
sh a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: lhu	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x03,0x55,0x00,0x01]
// CHECK-CAP-MODE-DISASM-NEXT: 01005503 <unknown>
lhu a0, 16(zero) // expected-error:12 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: lw	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x03,0x25,0x00,0x01]
// CHECK-CAP-MODE-DISASM-NEXT: 01002503 <unknown>
lw a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: sw	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x23,0x28,0xa0,0x00]
// CHECK-CAP-MODE-DISASM-NEXT: 00a02823 <unknown>
sw a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: lw	a0, 0(zero)
// CHECK-ASM-SAME: # encoding: [0x03,0x25,0x00,0x00]
// CHECK-CAP-MODE-DISASM-NEXT: 00002503 <unknown>
lw a0, (zero) // expected-error:9 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: sw	a0, 0(zero)
// CHECK-ASM-SAME: # encoding: [0x23,0x20,0xa0,0x00]
// CHECK-CAP-MODE-DISASM-NEXT: 00a02023 <unknown>
sw a0, (zero) // expected-error:9 {{register must be a GPR excluding zero (x0)}}
.ifdef RV64
// CHECK-64: lwu	a0, 16(zero)
// CHECK-ASM-64-SAME: # encoding: [0x03,0x65,0x00,0x01]
// CHECK-CAP-MODE-DISASM-64-NEXT: 01006503 <unknown>
lwu a0, 16(zero) // rv64-error:12 {{register must be a GPR excluding zero (x0)}}
// CHECK-64: ld	a0, 16(zero)
// CHECK-ASM-64-SAME: # encoding:  [0x03,0x35,0x00,0x01]
// CHECK-CAP-MODE-DISASM-64-NEXT: 01003503 <unknown>
ld a0, 16(zero) // rv64-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-64: sd	a0, 16(zero)
// CHECK-ASM-64-SAME: # encoding: [0x23,0x38,0xa0,0x00]
// CHECK-CAP-MODE-DISASM-64-NEXT: 00a03823 <unknown>
sd a0, 16(zero) // rv64-error:11 {{register must be a GPR excluding zero (x0)}}
.endif
// CHECK-NEXT: ly	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x7b,0x15,0x00,0x01]
// CHECK-CAP-MODE-DISASM-NEXT: 0100157b <unknown>
ly a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
// CHECK-NEXT: sy	a0, 16(zero)
// CHECK-ASM-SAME: # encoding: [0x7b,0x28,0xa0,0x00]
// CHECK-CAP-MODE-DISASM-NEXT: 00a0287b <unknown>
sy a0, 16(zero) // expected-error:11 {{register must be a GPR excluding zero (x0)}}
