# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

# Test some variations of padding to the end of a bundle.

  .text
foo:
  .bundle_align_mode 4

# Each of these callq instructions is 5 bytes long
  callq   bar
  callq   bar
  .bundle_lock align_to_end
  callq   bar
  .bundle_unlock
# To align this group to a bundle end, we need a 1-byte NOP.
# CHECK:        a:  nop
# CHECK-NEXT:   b: callq

  callq   bar
  callq   bar
  .bundle_lock align_to_end
  callq   bar
  callq   bar
  .bundle_unlock
# Here we have to pad until the end of the *next* boundary because
# otherwise the group crosses a boundary.
# CHECK:      1a: nop
# The nop sequence may be implemented as one instruction or many, but if
# it's one instruction, that instruction cannot itself cross the boundary.
# CHECK:      20: nop
# CHECK-NEXT: 26: callq
# CHECK-NEXT: 2b: callq

  .p2align 4
lock_to_next_bundle:
  .rept 14
  inc %eax
  .endr
  .bundle_lock align_to_end
  inc %eax
  inc %eax
  inc %eax
  .bundle_unlock
# This bundle group must be adjusted to the next boundary,
# but nop optimization must not break the bundle too.
# CHECK:      30: inc
# CHECK:      4a: inc
# CHECK-NEXT: 4c: nop
# CHECK-NEXT: 50: nop
# CHECK-NEXT: 5a: inc

lock_just_fit:
  .rept 14
  inc %eax
  .endr
  .bundle_lock align_to_end
  inc %eax
  inc %eax
  .bundle_unlock
# CHECK:      60: inc
# CHECK:      7e: inc

