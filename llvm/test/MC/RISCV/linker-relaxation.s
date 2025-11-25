# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=NORELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=NORELAX-RELOC %s

.long foo

call foo
# NORELAX-RELOC: R_RISCV_CALL_PLT foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_CALL_PLT foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

lui t1, %hi(foo)
# NORELAX-RELOC: R_RISCV_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

addi t1, t1, %lo(foo)
# NORELAX-RELOC: R_RISCV_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

sb t1, %lo(foo)(a2)
# NORELAX-RELOC: R_RISCV_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

1:
auipc t1, %pcrel_hi(foo)
# NORELAX-RELOC: R_RISCV_PCREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

addi t1, t1, %pcrel_lo(1b)
# NORELAX-RELOC: R_RISCV_PCREL_LO12_I .Ltmp0 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_I .Ltmp0 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

sb t1, %pcrel_lo(1b)(a2)
# NORELAX-RELOC: R_RISCV_PCREL_LO12_S .Ltmp0 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_S .Ltmp0 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0


# Check behaviour when a locally defined symbol is referenced.
bar:

beq s1, s1, bar
# NORELAX-RELOC-NOT: R_RISCV_BRANCH

call bar
# NORELAX-RELOC-NOT: R_RISCV_CALL
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC-NEXT: R_RISCV_CALL_PLT bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

beq s1, s1, bar
# NORELAX-RELOC-NOT: R_RISCV_BRANCH
# RELAX-RELOC-NEXT: R_RISCV_BRANCH bar 0x0

lui t1, %hi(bar)
# NORELAX-RELOC: R_RISCV_HI20 bar 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_HI20 bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

addi t1, t1, %lo(bar)
# NORELAX-RELOC: R_RISCV_LO12_I bar 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_I bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

sb t1, %lo(bar)(a2)
# NORELAX-RELOC: R_RISCV_LO12_S bar 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_LO12_S bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

2:
auipc t1, %pcrel_hi(bar)
# NORELAX-RELOC-NOT: R_RISCV_PCREL_HI20
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_HI20 bar 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

addi t1, t1, %pcrel_lo(2b)
# NORELAX-RELOC-NOT: R_RISCV_PCREL_LO12_I
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_I .Ltmp1 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

sb t1, %pcrel_lo(2b)(a2)
# NORELAX-RELOC-NOT: R_RISCV_PCREL_LO12_S
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_PCREL_LO12_S .Ltmp1 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0

## %hi/%lo on an absolute symbol (not yet defined) leads to relocations when relaxation is enabled.
lui t2, %hi(abs)
# NORELAX-RELOC-NOT: R_RISCV_
# RELAX-RELOC:      R_RISCV_HI20 - 0x12345
# RELAX-RELOC-NEXT: R_RISCV_RELAX - 0x0

addi t2, t2, %lo(abs)
# NORELAX-RELOC-NOT: R_RISCV_
# RELAX-RELOC:      R_RISCV_LO12_I - 0x12345
# RELAX-RELOC-NEXT: R_RISCV_RELAX - 0x0

.set abs, 0x12345

lui t3, %hi(abs)
# RELAX-RELOC:      R_RISCV_HI20 - 0x12345
# RELAX-RELOC-NEXT: R_RISCV_RELAX - 0x0

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
