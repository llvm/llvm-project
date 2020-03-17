  # RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --x86-align-branch-boundary=32 --x86-align-branch=jmp -x86-pad-max-prefix-size=5 %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
  # Check instructions can be aligned correctly along with option -x86-pad-max-prefix-size=5

  .text
  .p2align 5
  .rept 24
  int3
  .endr
  # We should not increase the length of this jmp to reduce the bytes of
  # following nops, doing so would make the jmp misaligned.
# CHECK:      18:          jmp
  jmp bar
# CHECK:      1d:          nop
# CHECK:      1e:          nop
# CHECK:      1f:          nop
# CHECK:      20:          int3
  .p2align 5
  int3
