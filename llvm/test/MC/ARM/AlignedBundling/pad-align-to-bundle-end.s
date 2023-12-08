# RUN: llvm-mc -filetype=obj -triple armv7-linux-gnueabi %s -o - \
# RUN:   | llvm-objdump --no-show-raw-insn --triple=armv7 -d - | FileCheck %s

	.syntax unified
	.text
  .bundle_align_mode 4

  bx lr
  and r1, r1, r2
  and r1, r1, r2
  .bundle_lock align_to_end
  bx r9
  .bundle_unlock
# No padding required here because bx just happens to be in the
# right offset.
# CHECK:      8:  and
# CHECK-NEXT: c:  bx

  bx lr
  and r1, r1, r2
  .bundle_lock align_to_end
  bx r9
  .bundle_unlock
# A 4-byte padding is needed here
# CHECK:      18: nop
# CHECK-NEXT: 1c: bx

  bx lr
  and r1, r1, r2
  .bundle_lock align_to_end
  bx r9
  bx r9
  bx r9
  .bundle_unlock
# A 12-byte padding is needed here to push the group to the end of the next
# bundle
# CHECK:      28: nop
# CHECK-NEXT: 2c: nop
# CHECK-NEXT: 30: nop
# CHECK-NEXT: 34: bx

