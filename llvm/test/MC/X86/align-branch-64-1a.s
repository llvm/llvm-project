# Check only fused conditional jumps, conditional jumps and unconditional jumps are aligned with option --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+jmp
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+jmp %s | llvm-objdump -d  - > %t1
# RUN: FileCheck --input-file=%t1 %s

# Check no branches is aligned with option --x86-align-branch-boundary=0
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=0 --x86-align-branch=fused+jcc+jmp %s | llvm-objdump -d  - > %t2
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s | llvm-objdump -d  - > %t3
# RUN: cmp %t2 %t3

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            18: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       1b: 31 c0                            xorl    %eax, %eax
# CHECK-COUNT-3:      : 90                               nop
# CHECK:            20: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       23: 74 5d                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            3d: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       3f: 90                               nop
# CHECK-NEXT:       40: 74 40                            je      {{.*}}
# CHECK-NEXT:       42: 5d                               popq    %rbp
# CHECK-NEXT:       43: 74 3d                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            5d: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       5f: 90                               nop
# CHECK-NEXT:       60: eb 26                            jmp     {{.*}}
# CHECK-NEXT:       62: eb 24                            jmp     {{.*}}
# CHECK-NEXT:       64: eb 22                            jmp     {{.*}}
# CHECK-COUNT-2:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            76: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK-NEXT:       79: 5d                               popq    %rbp
# CHECK-NEXT:       7a: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       7d: 74 03                            je      {{.*}}
# CHECK-NEXT:       7f: 90                               nop
# CHECK-NEXT:       80: eb 06                            jmp     {{.*}}
# CHECK-NEXT:       82: 8b 45 f4                         movl    -12(%rbp), %eax
# CHECK-NEXT:       85: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK-COUNT-10:     : 89 b5 50 fb ff ff                movl    %esi, -1200(%rbp)
# CHECK:            c4: eb c2                            jmp     {{.*}}
# CHECK-NEXT:       c6: c3                               retq

  .text
  .globl  foo
  .p2align  4
foo:
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  cmp  %rax, %rbp
  xorl %eax, %eax
  cmp  %rax, %rbp
  je  .L_2
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  xorl %eax, %eax
  je  .L_2
  popq  %rbp
  je  .L_2
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  xorl %eax, %eax
  jmp  .L_3
  jmp  .L_3
  jmp  .L_3
  .rept 2
  movl  %eax, %fs:0x1
  .endr
  movl  %eax, -4(%rbp)
  popq  %rbp
  cmp  %rax, %rbp
  je  .L_2
  jmp  .L_3
.L_2:
  movl  -12(%rbp), %eax
  movl  %eax, -4(%rbp)
.L_3:
  .rept 10
  movl  %esi, -1200(%rbp)
  .endr
  jmp  .L_3
  retq
