## Test that long branches are not relaxed with -riscv-asm-relax-branches=0
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c \
# RUN:       -riscv-asm-relax-branches=0 %t/pass.s \
# RUN:     | llvm-objdump -dr -M no-aliases - \
# RUN:     | FileCheck %t/pass.s
# RUN: not llvm-mc -filetype=obj -triple=riscv64 -mattr=+c -o /dev/null \
# RUN:       -riscv-asm-relax-branches=0 %t/fail.s 2>&1 \
# RUN:     | FileCheck %t/fail.s

#--- pass.s
  .text
test_undefined:
## Branches to undefined symbols should not be relaxed
# CHECK:      bne a0, a1, {{.*}}
# CHECK-NEXT: R_RISCV_BRANCH foo
   bne a0, a1, foo
# CHECK:      c.beqz a0, {{.*}}
# CHECK-NEXT: R_RISCV_RVC_BRANCH foo
   c.beqz a0, foo
# CHECK:      c.j {{.*}}
# CHECK-NEXT: R_RISCV_RVC_JUMP foo
   c.j foo

## Branches to defined in-range symbols should work normally
test_defined_in_range:
# CHECK:      bne a0, a1, {{.*}} <bar>
  bne a0, a1, bar
# CHECK:      c.beqz a0, {{.*}} <bar>
   c.beqz a0, bar
# CHECK:      c.j {{.*}} <bar>
   c.j bar
bar:

#--- fail.s
  .text
## Branches to defined out-of-range symbols should report an error
test_defined_out_of_range:
  bne a0, a1, 1f # CHECK: :[[#@LINE]]:3: error: fixup value out of range
  .skip (1 << 12)
1:
  c.beqz a0, 1f # CHECK: :[[#@LINE]]:3: error: fixup value out of range
  .skip (1 << 8)
1:
  c.j 1f # CHECK: :[[#@LINE]]:3: error: fixup value out of range
  .skip (1 << 11)
1:
