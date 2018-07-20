# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro %s -o - \
# RUN:   | llvm-objdump -disassemble -no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -disassemble -no-show-raw-insn - | FileCheck %s

# Test that long nops are generated for padding where possible.

  .text
foo:
  .bundle_align_mode 5

# This callq instruction is 5 bytes long
  .bundle_lock align_to_end
  callq   bar
  .bundle_unlock
# To align this group to a bundle end, we need a two 10-byte NOPs and a 7-byte NOP.
# CHECK:        0:  nop
# CHECK-NEXT:   a:  nop
# CHECK-NEXT:   14: nop
# CHECK:   1b: callq

# This push instruction is 1 byte long
  .bundle_lock align_to_end
  push %rax
  .bundle_unlock
# To align this group to a bundle end, we need three 10-byte NOPs, and a 1-byte.
# CHECK:        20:  nop
# CHECK-NEXT:   2a:  nop
# CHECK-NEXT:   34:  nop
# CHECK-NEXT:   3e:  nop
# CHECK-NEXT:   3f: pushq
