# RUN: llvm-mc -triple riscv32 -show-encoding -mattr=+c,+relax \
# RUN:   -M no-aliases %s | FileCheck -check-prefixes=CHECK-ASM,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -filetype=obj -mattr=+c,+relax %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c -dr --no-print-imm-hex -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-OBJDUMP,CHECK-INST %s

# RUN: llvm-mc -triple riscv64 -show-encoding \
# RUN:   -M no-aliases -mattr=+c,+relax %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -triple riscv64 -filetype=obj -mattr=+c,+relax %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c -dr --no-print-imm-hex -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-OBJDUMP,CHECK-INST %s

## `.option exact` disables a variety of assembler behaviour:
## - automatic compression
## - branch relaxation (of short branches to longer equivalent sequences)
## - linker relaxation (emitting R_RISCV_RELAX)
## `.option noexact` enables these behaviours again. It is also the default.

# CHECK-OBJDUMP: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ASM: # encoding: [0x08,0x41]
lw a0, 0(a0)

# CHECK-OBJDUMP: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ASM: # encoding: [0x08,0x41]
c.lw a0, 0(a0)

# CHECK-ASM: call undefined
# CHECK-ASM-SAME: # encoding: [0x97'A',A,A,A,0xe7'A',0x80'A',A,A]
# CHECK-OBJDUMP: auipc ra, 0
# CHECK-OBJDUMP-NEXT: R_RISCV_CALL_PLT undefined
# CHECK-OBJDUMP-NEXT: R_RISCV_RELAX *ABS*
# CHECK-OBJDUMP-NEXT: jalr ra
call undefined@plt

# CHECK-ASM: beq a0, a1, undefined
# CHECK-ASM-SAME: # encoding: [0x63'A',A,0xb5'A',A]
# CHECK-OBJDUMP: bne a0, a1, 0x14
# CHECK-OBJDUMP-NEXT: jal zero, 0x10
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
beq a0, a1, undefined

# CHECK-ASM: c.j undefined
# CHECK-ASM-SAME: # encoding: [0bAAAAAA01,0b101AAAAA]
# CHECK-OBJDUMP: jal zero, 0x14
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
c.j undefined

# CHECK-ASM: .option exact
.option exact

# CHECK-OBJDUMP: 00052503
# CHECK-INST: lw a0, 0(a0)
# CHECK-ASM: # encoding: [0x03,0x25,0x05,0x00]
lw a0, 0(a0)

# CHECK-OBJDUMP: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ASM: # encoding: [0x08,0x41]
c.lw a0, 0(a0)

# CHECK-ASM: call undefined
# CHECK-ASM-SAME: # encoding: [0x97'A',A,A,A,0xe7'A',0x80'A',A,A]
# CHECK-OBJDUMP: auipc ra, 0
# CHECK-OBJDUMP-NEXT: R_RISCV_CALL_PLT undefined
# CHECK-OBJDUMP-NOT: R_RISCV_RELAX
# CHECK-OBJDUMP-NEXT: jalr ra, 0(ra)
call undefined@plt

# CHECK-ASM: beq a0, a1, undefined
# CHECK-ASM-SAME: # encoding: [0x63'A',A,0xb5'A',A]
# CHECK-OBJDUMP: beq a0, a1, 0x26
# CHECK-OBJDUMP-NEXT: R_RISCV_BRANCH undefined
beq a0, a1, undefined

# CHECK-ASM: c.j undefined
# CHECK-ASM-SAME: # encoding: [0bAAAAAA01,0b101AAAAA]
# CHECK-OBJDUMP: c.j 0x2a
# CHECK-OBJDUMP-NEXT: R_RISCV_RVC_JUMP undefined
c.j undefined

# CHECK-ASM: .option noexact
.option noexact

# CHECK-OBJDUMP: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ASM: # encoding: [0x08,0x41]
lw a0, 0(a0)

# CHECK-OBJDUMP: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ASM: # encoding: [0x08,0x41]
c.lw a0, 0(a0)

# CHECK-ASM: call undefined
# CHECK-ASM-SAME: # encoding: [0x97'A',A,A,A,0xe7'A',0x80'A',A,A]
# CHECK-OBJDUMP: auipc ra, 0
# CHECK-OBJDUMP-NEXT: R_RISCV_CALL_PLT undefined
# CHECK-OBJDUMP-NEXT: R_RISCV_RELAX *ABS*
# CHECK-OBJDUMP-NEXT: jalr ra, 0(ra)
call undefined@plt

# CHECK-ASM: beq a0, a1, undefined
# CHECK-ASM-SAME: # encoding: [0x63'A',A,0xb5'A',A]
# CHECK-OBJDUMP: bne a0, a1, 0x40
# CHECK-OBJDUMP-NEXT: jal zero, 0x3c
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
beq a0, a1, undefined

# CHECK-ASM: c.j undefined
# CHECK-ASM-SAME: # encoding: [0bAAAAAA01,0b101AAAAA]
# CHECK-OBJDUMP: jal zero, 0x40
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
c.j undefined
