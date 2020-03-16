# RUN: llvm-mc -filetype=obj -triple armv7-linux-gnueabi %s -o - \
# RUN:   | llvm-objdump --no-show-raw-insn --triple=armv7 -d - | FileCheck %s

# On ARM each instruction is 4 bytes long so padding for individual
# instructions should not be inserted. However, for bundle-locked groups
# it can be.

  .syntax unified
  .text
  .bundle_align_mode 4

  bx lr
  and r1, r1, r2
  and r1, r1, r2
  .bundle_lock
  bx r9
  bx r8
  .bundle_unlock
# CHECK:      c:  nop
# CHECK-NEXT: 10: bx
# CHECK-NEXT: 14: bx

  # pow2 here
  .align 4 
  bx lr
  .bundle_lock
  bx r9
  bx r9
  bx r9
  bx r8
  .bundle_unlock
# CHECK:      20: bx
# CHECK-NEXT: 24: nop
# CHECK-NEXT: 28: nop
# CHECK-NEXT: 2c: nop
# CHECK-NEXT: 30: bx

  .align 4
foo:
  b foo
  .long 3892240112
  .long 3892240112
  .long 3892240112
  .long 3892240112
  .long 3892240112
  .long 3892240112
# CHECK:  40: b

