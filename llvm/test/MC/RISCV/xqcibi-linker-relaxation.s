# RUN: llvm-mc --triple=riscv32 -mattr=+relax,+experimental-xqcilb,+experimental-xqcibi \
# RUN:    %s -filetype=obj -o - -riscv-add-build-attributes \
# RUN:    | llvm-objdump -dr -M no-aliases - \
# RUN:    | FileCheck %s

## This tests that we correctly emit relocations for linker relaxation when
## relaxing `JAL` to `QC.E.JAL`.

## PR150071

.global foo

# CHECK-LABEL: <branch_over_relaxable>:
branch_over_relaxable:
  jal x1, foo
# CHECK: qc.e.jal 0x0 <branch_over_relaxable>
# CHECK-NEXT: R_RISCV_VENDOR QUALCOMM
# CHECK-NEXT: R_RISCV_CUSTOM195 foo
# CHECK-NEXT: R_RISCV_RELAX *ABS*
  bne a0, a1, branch_over_relaxable
# CHECK-NEXT: bne a0, a1, 0x6 <branch_over_relaxable+0x6>
# CHECK-NEXT: R_RISCV_BRANCH branch_over_relaxable
# CHECK-NOT: R_RISCV_RELAX
  qc.e.bnei a0, 0x21, branch_over_relaxable
# CHECK-NEXT: qc.e.bnei a0, 0x21, 0xa <branch_over_relaxable+0xa>
# CHECK-NEXT: R_RISCV_VENDOR QUALCOMM
# CHECK-NEXT: R_RISCV_CUSTOM193 branch_over_relaxable
# CHECK-NOT: R_RISCV_RELAX
  ret
# CHECK-NEXT: c.jr ra

# CHECK-LABEL: <short_jump_over_fixed>:
short_jump_over_fixed:
  nop
# CHECK: c.nop
  j short_jump_over_fixed
# CHECK-NEXT: c.j 0x12 <short_jump_over_fixed>
# CHECK-NOT: R_RISCV_RVC_JUMP
# CHECK-NOT: R_RISCV_RELAX
  ret
# CHECK-NEXT: c.jr ra

# CHECK-LABEL: <short_jump_over_relaxable>:
short_jump_over_relaxable:
  call foo
# CHECK: auipc ra, 0x0
# CHECK-NEXT: R_RISCV_CALL_PLT foo
# CHECK-NEXT: R_RISCV_RELAX *ABS*
# CHECK-NEXT: jalr ra, 0x0(ra) <short_jump_over_relaxable>
  j short_jump_over_relaxable
# CHECK-NEXT: c.j 0x20 <short_jump_over_relaxable+0x8>
# CHECK-NEXT: R_RISCV_RVC_JUMP short_jump_over_relaxable
# CHECK-NOT: R_RISCV_RELAX
  ret
# CHECK-NEXT: c.jr ra

# CHECK-LABEL: <mid_jump_over_fixed>:
mid_jump_over_fixed:
  nop
# CHECK: c.nop
  .space 0x1000
# CHECK-NEXT: ...
  j mid_jump_over_fixed
# CHECK-NEXT: jal zero, 0x24 <mid_jump_over_fixed>
# CHECK-NOT: R_RISCV_JAL
# CHECK-NOT: R_RISCV_RELAX
  ret
# CHECK-NEXT: c.jr ra

# CHECK-LABEL: <mid_jump_over_relaxable>:
mid_jump_over_relaxable:
  call foo
# CHECK: auipc ra, 0x0
# CHECK-NEXT: R_RISCV_CALL_PLT foo
# CHECK-NEXT: R_RISCV_RELAX *ABS*
# CHECK-NEXT: jalr ra, 0x0(ra) <mid_jump_over_relaxable>
  .space 0x1000
# CHECK-NEXT: ...
  j mid_jump_over_relaxable
# CHECK-NEXT: jal zero, 0x2034 <mid_jump_over_relaxable+0x1008>
# CHECK-NEXT: R_RISCV_JAL mid_jump_over_relaxable
# CHECK-NOT: R_RISCV_RELAX
  ret
# CHECK-NEXT: c.jr ra
