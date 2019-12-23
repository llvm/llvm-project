# Check only fused conditional jumps and conditional jumps are aligned with option --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc %p/Inputs/align-branch-64-1.s | llvm-objdump -d  - | FileCheck %s

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       18: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       1b: 31 c0                            xorl    %eax, %eax
# CHECK-COUNT-3:      : 90                               nop
# CHECK-NEXT:       20: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       23: 74 5b                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            3d: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       3f: 90                               nop
# CHECK-NEXT:       40: 74 3e                            je      {{.*}}
# CHECK-NEXT:       42: 5d                               popq    %rbp
# CHECK-NEXT:       43: 74 3b                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            5d: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       5f: eb 25                            jmp     {{.*}}
# CHECK-NEXT:       61: eb 23                            jmp     {{.*}}
# CHECK-NEXT:       63: eb 21                            jmp     {{.*}}
# CHECK-COUNT-2:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       75: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK:            78: 5d                               popq    %rbp
# CHECK-NEXT:       79: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       7c: 74 02                            je      {{.*}}
# CHECK-NEXT:       7e: eb 06                            jmp     {{.*}}
# CHECK-NEXT:       80: 8b 45 f4                         movl    -12(%rbp), %eax
# CHECK-NEXT:       83: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK-COUNT-10:     : 89 b5 50 fb ff ff                movl    %esi, -1200(%rbp)
# CHECK:            c2: eb c2                            jmp     {{.*}}
# CHECK-NEXT:       c4: c3                               retq
