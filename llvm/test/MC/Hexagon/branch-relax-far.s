# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# Test coverage for HexagonAsmBackend: exercise branch relaxation for
# short conditional branches (B9_PCREL, B13_PCREL, B15_PCREL) that need
# constant extenders when the target is too far away.

# A conditional branch with a far target should be relaxed with an
# extender (immext).

# CHECK-LABEL: <test_b15_relax>:
# CHECK: immext(
# CHECK: if (p0) jump:nt
.globl test_b15_relax
test_b15_relax:
  {
    if (p0) jump:nt .Lfar_target_b15
  }
  .space 70000
.Lfar_target_b15:
  jumpr lr

# CHECK-LABEL: <test_b13_relax>:
# CHECK: immext(
# CHECK: if (cmp.eq(r0.new,#0)) jump:nt
.globl test_b13_relax
test_b13_relax:
  {
    r0 = r1
    if (cmp.eq(r0.new, #0)) jump:nt .Lfar_target_b13
  }
  .space 10000
.Lfar_target_b13:
  jumpr lr
