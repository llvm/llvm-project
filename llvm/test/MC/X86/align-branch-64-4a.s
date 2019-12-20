# Check only rets are aligned with option --x86-align-branch-boundary=32 --x86-align-branch=ret
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=ret %s | llvm-objdump -d  - | FileCheck %s

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-COUNT-2:      : 48 89 e5                         movq    %rsp, %rbp
# CHECK:            1e: 5a                               popq    %rdx
# CHECK-NEXT:       1f: 90                               nop
# CHECK-NEXT:       20: c3                               retq
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            39: 89 75 f4                         movl    %esi, -12(%rbp)
# CHECK-NEXT:       3c: 31 c0                            xorl    %eax, %eax
# CHECK-COUNT-2:      : 90                               nop
# CHECK:            40: c2 1e 00                         retq    $30

  .text
  .globl  foo
  .p2align  4
foo:
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  .rept 2
  movq  %rsp, %rbp
  .endr
  popq  %rdx
  ret
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  movl  %esi, -12(%rbp)
  xorl %eax, %eax
  ret $30
