# RUN: llvm-mc %s -mc-relax-all -triple=riscv64 -filetype=obj \
# RUN:     | llvm-objdump -d -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK %s

# This test previously crashed because expanding a conditional branch deleted
# all fixups in the fragment.

# CHECK:      beq     s0, zero, 0x8
# CHECK-NEXT: jal     zero, 0x14
# CHECK-NEXT: jal     zero, 0x14
# CHECK-NEXT: bne     s0, zero, 0x14
# CHECK-NEXT: jal     zero, 0x14

# CHECK:      jalr    zero, 0x0(ra)
  bnez s0, .foo
  j    .foo
  beqz s0, .foo
.foo:
  ret
