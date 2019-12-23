# Check only conditional jumps are aligned with option --x86-align-branch-boundary=32 --x86-align-branch=jcc
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=jcc %p/Inputs/align-branch-64-1.s | llvm-objdump -d  - | FileCheck %s

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            18: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       1b: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       1d: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       20: 74 5b                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            3a: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       3c: 74 3f                            je      {{.*}}
# CHECK-NEXT:       3e: 5d                               popq    %rbp
# CHECK-NEXT:       3f: 90                               nop
# CHECK-NEXT:       40: 74 3b                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            5a: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       5c: eb 25                            jmp     {{.*}}
# CHECK-NEXT:       5e: eb 23                            jmp     {{.*}}
# CHECK-NEXT:       60: eb 21                            jmp     {{.*}}
# CHECK-COUNT-2:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            72: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK-NEXT:       75: 5d                               popq    %rbp
# CHECK-NEXT:       76: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       79: 74 02                            je      {{.*}}
# CHECK-NEXT:       7b: eb 06                            jmp     {{.*}}
# CHECK-NEXT:       7d: 8b 45 f4                         movl    -12(%rbp), %eax
# CHECK-NEXT:       80: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK-COUNT-10:     : 89 b5 50 fb ff ff                movl    %esi, -1200(%rbp)
# CHECK:            bf: eb c2                            jmp     {{.*}}
# CHECK-NEXT:       c1: c3                               retq
