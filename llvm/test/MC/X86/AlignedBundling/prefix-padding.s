# RUN: llvm-mc -filetype=obj -triple x86_64 %s --x86-pad-max-prefix-size=5 \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  .section text1, "x"
add_prefix_prev:
  .bundle_align_mode 5
## This callq instruction is 5 bytes long
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
  ## instructions inside a bundle lock can also be prefix-padded.
  callq   bar
  .bundle_unlock
# CHECK:        20: call
# CHECK-NEXT:   2a: nop
# CHECK:        36: call

  .p2align 5
ignore_nop_for_p2align:
  int3
  ## no prefix padding with this 15-byte nop.
  .p2align 4
  ## prefix padding with only callq, not the early int3
  .bundle_lock align_to_end
  callq bar
  .bundle_unlock
# CHECK:        40: int3
# CHECK-NEXT:   41: nop
# CHECK-NEXT:   50: nop
# CHECK-NEXT:   56: call

ignore_nop_for_p2align5:
  callq   bar
  .p2align 5
.L1:
  callq   bar
# CHECK:        60: call
# CHECK-NEXT:   65: nop
# CHECK:        80: call

  .section text2, "x"
## ensure the last instructions are not prefix-padded
tail_bundle:
  .bundle_lock
  callq   bar
  .bundle_unlock
  nop
# CHECK:        0: call
# CHECK-NEXT:   5: nop

## Without prefix padding, the align_to_end group creates a 24-byte nop that
## spans a bundle boundary, out of a single BoundaryAlign Fragment. This test
## case ensures the 24-byte nop is consumed without overflowing any bundle.
  .section text3, "x"
ensure_bundle_boundary:
  ## 20-byte group (5-byte * 4)
  .bundle_lock
  .rept 4
  callq foo
  .endr
  .bundle_unlock
  ## consume 12-byte by prefix-padding the first group (back-to-front).
  ## consume another 12-byte by prefix-padding the second group (front-to-back).
  .bundle_lock align_to_end
  ## Although a maximum prefix budget is 20 bytes from this group,
  ## only 12 bytes will be used.
  .rept 4
  callq bar
  .endr
  .bundle_unlock
# CHECK:        0:  call
# CHECK-NEXT:   5:  call
# CHECK-NEXT:   c:  call
# CHECK-NEXT:   16: call
# CHECK-NEXT:   20: call
# CHECK-NEXT:   2a: call
# CHECK-NEXT:   34: call
# CHECK-NEXT:   3b: call
