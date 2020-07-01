# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=16 --x86-align-branch=fused+jcc --mc-relax-all %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

# Check using option --x86-align-branch-boundary=16 --x86-align-branch=fused+jcc --mc-relax-all with bundle won't make code crazy

# CHECK:            0:       pushq    %rbp
# CHECK-NEXT:       1:       testq    $2, %rdx
# CHECK-NEXT:       8:       jne
# CHECK-NEXT:       e:       nop
# CHECK-NEXT:      10:       jle

    .text
    .p2align 4
foo:
  push %rbp
  # Will be bundle-aligning to 8 byte boundaries
  .bundle_align_mode 3
  test $2, %rdx
  jne   foo
# This jle is 6 bytes long and should have started at 0xe, so two bytes
# of nop padding are inserted instead and it starts at 0x10
  jle   foo
