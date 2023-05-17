# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=NORELAX-RELOC %s
# RUN: llvm-mc -triple riscv32 -mattr=+relax < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=RELAX-FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=NORELAX-RELOC %s
# RUN: llvm-mc -triple riscv64 -mattr=+relax < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=RELAX-FIXUP %s

.long foo

call foo
# NORELAX-RELOC: R_RISCV_CALL_PLT foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_CALL_PLT foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_call
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lui t1, %hi(foo)
# NORELAX-RELOC: R_RISCV_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %hi(foo), kind: fixup_riscv_hi20
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

addi t1, t1, %lo(foo)
# NORELAX-RELOC: R_RISCV_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %lo(foo), kind: fixup_riscv_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

sb t1, %lo(foo)(a2)
# NORELAX-RELOC: R_RISCV_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %lo(foo), kind: fixup_riscv_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

1:
auipc t1, %pcrel_hi(foo)
# NORELAX-RELOC: R_RISCV_PCREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %pcrel_hi(foo), kind: fixup_riscv_pcrel_hi20
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

addi t1, t1, %pcrel_lo(1b)
# NORELAX-RELOC: R_RISCV_PCREL_LO12_I .Ltmp0 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_I .Ltmp0 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %pcrel_lo(.Ltmp0), kind: fixup_riscv_pcrel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

sb t1, %pcrel_lo(1b)(a2)
# NORELAX-RELOC: R_RISCV_PCREL_LO12_S .Ltmp0 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_S .Ltmp0 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %pcrel_lo(.Ltmp0), kind: fixup_riscv_pcrel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax


# Check behaviour when a locally defined symbol is referenced.
bar:

beq s1, s1, bar
# NORELAX-RELOC-NOT: R_RISCV_BRANCH
# RELAX-RELOC: R_RISCV_BRANCH bar 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: bar, kind: fixup_riscv_branch
# RELAX-FIXUP-NOT: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

call bar
# NORELAX-RELOC-NOT: R_RISCV_CALL
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_CALL_PLT bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: bar, kind: fixup_riscv_call
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lui t1, %hi(bar)
# NORELAX-RELOC: R_RISCV_HI20 bar 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_HI20 bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %hi(bar), kind: fixup_riscv_hi20
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

addi t1, t1, %lo(bar)
# NORELAX-RELOC: R_RISCV_LO12_I bar 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_I bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %lo(bar), kind: fixup_riscv_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

sb t1, %lo(bar)(a2)
# NORELAX-RELOC: R_RISCV_LO12_S bar 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_S bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %lo(bar), kind: fixup_riscv_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

2:
auipc t1, %pcrel_hi(bar)
# NORELAX-RELOC-NOT: R_RISCV_PCREL_HI20
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_HI20 bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %pcrel_hi(bar), kind: fixup_riscv_pcrel_hi20
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

addi t1, t1, %pcrel_lo(2b)
# NORELAX-RELOC-NOT: R_RISCV_PCREL_LO12_I
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_I .Ltmp1 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %pcrel_lo(.Ltmp1), kind: fixup_riscv_pcrel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

sb t1, %pcrel_lo(2b)(a2)
# NORELAX-RELOC-NOT: R_RISCV_PCREL_LO12_S
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_S .Ltmp1 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %pcrel_lo(.Ltmp1), kind: fixup_riscv_pcrel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

# Check that a relocation is not emitted for a symbol difference which has
# been folded to a fixup with an absolute value. This can happen when a
# difference expression refers to two symbols, at least one of which is
# not defined at the point it is referenced. Then during *assembler*
# relaxation when both symbols have become defined the difference may be folded
# down to a fixup simply containing the absolute value. We want to ensure that
# we don't force a relocation to be emitted for this absolute value even
# when linker relaxation is enabled. The reason for this is that one instance
# where this pattern appears in in the .eh_frame section (the CIE 'length'
# field), and the .eh_frame section cannot be parsed by the linker unless the
# fixup has been resolved to a concrete value instead of a relocation.
  .data
lo:
  .word hi-lo
  .quad hi-lo
# NORELAX-RELOC-NOT: R_RISCV_32
# NORELAX-RELOC-NOT: R_RISCV_64
# RELAX-RELOC-NOT: R_RISCV_32
# RELAX-RELOC-NOT: R_RISCV_64
hi:
