# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64 -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

## Test NOP padding to remove every instruction that crosses a bundle boundary.

  .text
foo:
  .bundle_align_mode 5
  ## 5 bytes * 6 = 30 bytes
  callq   bar
  callq   bar
  callq   bar
  callq   bar
  callq   bar
  callq   bar

## This imull is 3 bytes long and should have started at 0x1d, so two bytes
## of nop padding are inserted instead and it starts at 0x20
  imull   $17, %ebx, %ebp
# CHECK:          1e: nop
# CHECK-NEXT:     20: imull

## Sub-bundle .align with single-instruction bundling:
## .align 16 is narrower than the 32-byte bundle; instructions after it
## start mid-bundle and still receive NOP padding as needed.
  pushq   %rbp
  .align  16
  movl $1, (%rsp)   ## 7 bytes at offset 16 — no padding needed
  movl $2, 4(%rsp)  ## 8 bytes; offset 23 → 23+8=31, fits in bundle
  movl $3, (%rsp)   ## 7 bytes at offset 31 → crosses boundary, pad to 32
# CHECK:      30: mov
# CHECK-NEXT: 37: mov
# CHECK-NEXT: 3f: nop
# CHECK-NEXT: 40: mov
