# RUN: llvm-mc -triple riscv32 -show-encoding \
# RUN:   -M no-aliases -mattr=+c,+relax %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -triple riscv32 -filetype=obj -mattr=+c,+relax %s \
# RUN:   | llvm-objdump  --triple=riscv32 -M no-aliases -dr - \
# RUN:   | FileCheck -check-prefixes=CHECK-OBJDUMP %s

# RUN: llvm-mc -triple riscv64 -show-encoding \
# RUN:   -M no-aliases -mattr=+c,+relax %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -triple riscv64 -filetype=obj -mattr=+c,+relax %s \
# RUN:   | llvm-objdump  --triple=riscv64  -M no-aliases -dr - \
# RUN:   | FileCheck -check-prefixes=CHECK-OBJDUMP %s

## `.option exact` disables a variety of assembler behaviour:
## - automatic compression
## - branch relaxation (of short branches to longer equivalent sequences)
## - linker relaxation (emitting R_RISCV_RELAX)
## `.option noexact` enables these behaviours again. It is also the default.

## This test only checks the branch and linker relaxation part of this behaviour.


# CHECK-ASM: call undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_call_plt
# CHECK-ASM-NEXT: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax
# CHECK-OBJDUMP: auipc ra, 0x0
# CHECK-OBJDUMP-NEXT: R_RISCV_CALL_PLT undefined
# CHECK-OBJDUMP-NEXT: R_RISCV_RELAX *ABS*
# CHECK-OBJDUMP-NEXT: jalr ra
call undefined@plt

# CHECK-ASM: beq a0, a1, undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_branch
# CHECK-OBJDUMP: bne a0, a1, 0x10
# CHECK-OBJDUMP-NEXT: jal zero, 0xc
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
beq a0, a1, undefined

# CHECK-ASM: c.j undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_rvc_jump
# CHECK-OBJDUMP: jal zero, 0x10
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
c.j undefined

# CHECK-ASM: .option exact
.option exact

# CHECK-ASM: call undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_call_plt
# CHECK-ASM-NOT: fixup_riscv_relax
# CHECK-OBJDUMP: auipc ra, 0x0
# CHECK-OBJDUMP-NEXT: R_RISCV_CALL_PLT undefined
# CHECK-OBJDUMP-NOT: R_RISCV_RELAX
# CHECK-OBJDUMP-NEXT: jalr ra
call undefined@plt

# CHECK-ASM: beq a0, a1, undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_branch
# CHECK-OBJDUMP: beq a0, a1, 0x1c
# CHECK-OBJDUMP-NEXT: R_RISCV_BRANCH undefined
beq a0, a1, undefined

# CHECK-ASM: c.j undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_rvc_jump
# CHECK-OBJDUMP: c.j 0x20
# CHECK-OBJDUMP-NEXT: R_RISCV_RVC_JUMP undefined
c.j undefined

# CHECK-ASM: .option noexact
.option noexact

# CHECK-ASM: call undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_call_plt
# CHECK-ASM-NEXT: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax
# CHECK-OBJDUMP: auipc ra, 0x0
# CHECK-OBJDUMP-NEXT: R_RISCV_CALL_PLT undefined
# CHECK-OBJDUMP-NEXT: R_RISCV_RELAX *ABS*
# CHECK-OBJDUMP-NEXT: jalr ra
call undefined@plt

# CHECK-ASM: beq a0, a1, undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_branch
# CHECK-OBJDUMP: bne a0, a1, 0x32
# CHECK-OBJDUMP-NEXT: jal zero, 0x2e
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
beq a0, a1, undefined

# CHECK-ASM: c.j undefined
# CHECK-ASM-NEXT: fixup A - offset: 0, value: undefined, kind: fixup_riscv_rvc_jump
# CHECK-OBJDUMP: jal zero, 0x32
# CHECK-OBJDUMP-NEXT: R_RISCV_JAL undefined
c.j undefined
