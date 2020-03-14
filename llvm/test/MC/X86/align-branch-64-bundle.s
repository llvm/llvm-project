# Check using option --x86-align-branch-boundary=16 --x86-align-branch=fused+jcc --mc-relax-all with bundle won't make code crazy
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=16 --x86-align-branch=fused+jcc --mc-relax-all %s | llvm-objdump -d  - > %t1
# RUN: FileCheck --input-file=%t1 %s

# CHECK: 0000000000000000 <foo>:
# CHECK-NEXT:       0: 55                               pushq    %rbp
# CHECK-NEXT:       1: 48 f7 c2 02 00 00 00             testq    $2, %rdx
# CHECK-NEXT:       8: 0f 85 f2 ff ff ff                jne      {{.*}}
# CHECK-NEXT:       e: 90                               nop
# CHECK-NEXT:       f: 90                               nop
# CHECK-NEXT:      10: 0f 8e ea ff ff ff                jle      {{.*}}

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
