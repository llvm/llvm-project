# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro %s -o - --x86-prefix-pad-for-bundle=1 --x86-pad-max-prefix-size=5 \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  .text
add_prefix_prev:
  .bundle_align_mode 5
# This callq instruction is 5 bytes long
  callq   bar
  callq   bar
  callq   bar
  callq   bar
  .bundle_lock align_to_end
  callq   bar
  .bundle_unlock
# CHECK:        0:  call
# CHECK-NEXT:   5:  call
# CHECK-NEXT:   a:  call
# CHECK-NEXT:   11:  call
# CHECK-NEXT:   1b:  call

  .p2align 5
add_prefix_prev_next:
  callq   bar
  .bundle_lock align_to_end
  # instructions inside a bundle lock can also be prefix-padded.
  callq   bar
  .bundle_unlock
# CHECK:        20: call
# CHECK-NEXT:   2a: nop
# CHECK:        36: call

  .p2align 5
ignore_nop_for_p2align:
  int3
  int3
  # no prefix padding with this 14-byte nop.
  .p2align 4
  int3
  .bundle_lock
  int3
  .bundle_unlock
  int3
# CHECK:        40: int3
# CHECK-NEXT:   41: int3
# CHECK-NEXT:   42: nop
# CHECK:        50: int3
# CHECK-NEXT:   51: int3
# CHECK-NEXT:   52: int3
# CHECK-NEXT:   53: nop

  .p2align 5
ignore_nop_for_p2align5:
  callq   bar
  .p2align 5
.L1:
  callq   bar
# CHECK:        60: call
# CHECK-NEXT:   65: nop
# CHECK:        80: call

# ensure the last instructions are not prefix-padded
  .p2align 5
tail_bundle:
  .bundle_lock
  callq   bar
  .bundle_unlock
  nop
# CHECK:        a0: call
# CHECK-NEXT:   a5: nop
