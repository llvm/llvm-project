# RUN: llvm-mc -filetype=obj -triple x86_64 %s \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=SHORT
# RUN: llvm-mc -filetype=obj -triple x86_64 -mc-relax-all %s \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=RELAX

  .text
  .bundle_align_mode 4
foo:
  .rept 12
  int3
  .endr
  jle .Lskip
.Lskip:
  int3

## Without relaxation the jump is 2 bytes at offset 0xc and does not cross the
## 0x10 boundary, so no padding is inserted.
# SHORT:      b: int3
# SHORT-NEXT: c: jle
# SHORT-NEXT: e: int3

## With -mc-relax-all the jump is 6 bytes, so a 4-byte NOP moves it to the
## next bundle at 0x10.
# RELAX:      b: int3
# RELAX-NEXT: c: nop
# RELAX-NEXT: 10: jle
# RELAX-NEXT: 16: int3
