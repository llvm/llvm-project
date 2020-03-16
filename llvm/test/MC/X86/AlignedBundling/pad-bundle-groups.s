# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

# Test some variations of padding for bundle-locked groups.

  .text
foo:
  .bundle_align_mode 4

# Each of these callq instructions is 5 bytes long
  callq   bar
  callq   bar

  .bundle_lock
  callq   bar
  callq   bar
  .bundle_unlock
# We'll need a 6-byte NOP before this group
# CHECK:        a:  nop
# CHECK-NEXT:   10: callq
# CHECK-NEXT:   15: callq

  .bundle_lock
  callq   bar
  callq   bar
  .bundle_unlock
# Same here
# CHECK:        1a:  nop
# CHECK-NEXT:   20: callq
# CHECK-NEXT:   25: callq

  .align 16, 0x90
  callq   bar
  .bundle_lock
  callq   bar
  callq   bar
  callq   bar
  .bundle_unlock
# And here we'll need a 10-byte NOP + 1-byte NOP
# CHECK:        30: callq
# CHECK:        35: nop
# CHECK:        3f: nop
# CHECK-NEXT:   40: callq
# CHECK-NEXT:   45: callq



