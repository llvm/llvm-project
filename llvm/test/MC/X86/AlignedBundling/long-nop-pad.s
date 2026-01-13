# RUN: llvm-mc -filetype=obj -triple x86_64 --mattr=+fast-15bytenop %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

## Test that long nops are generated for padding where possible while each respects the bundle align boundary.

  .text
foo:
  .bundle_align_mode 5

## This callq instruction is 5 bytes long
  .bundle_lock align_to_end
  callq   bar
  .bundle_unlock
## To align this group to a bundle end, we need a 15-byte NOPs and a 12-byte NOP.
# CHECK:        0:  nop
# CHECK-NEXT:   f:  nop
# CHECK-NEXT:   1b: callq

## This push instruction is 1 byte long
  .bundle_lock align_to_end
  push %rax
  .bundle_unlock
## To align this group to a bundle end, we need two 15-byte NOPs and a 1-byte NOP.
# CHECK:        20:  nop
# CHECK-NEXT:   2f:  nop
# CHECK-NEXT:   3e:  nop
# CHECK-NEXT:   3f:  pushq

## bundle-aware optimization for `.nops N`
  .p2align 5
just_nops:
  callq   bar
  .nops 64
# CHECK:        40:  callq
# CHECK-NEXT:   45:  nop
# CHECK-NEXT:   54:  nop
# CHECK-NEXT:   60:  nop
# CHECK-NEXT:   6f:  nop
# CHECK-NEXT:   7e:  nop
# CHECK-NEXT:   80:  nop
