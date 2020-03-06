# Check only conditional jumps and unconditional jumps are aligned with option --x86-align-branch-boundary=32 --x86-align-branch=jcc+jmp
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=jcc+jmp %p/Inputs/align-branch-64-1.s | llvm-objdump -d  - > %t1
# RUN: FileCheck --input-file=%t1 %s --check-prefixes=CHECK,SHORT-NOP

# Check long NOP can be emitted to align branch if the target cpu support long nop.
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 -mcpu=x86-64 --x86-align-branch=jcc+jmp %p/Inputs/align-branch-64-1.s | llvm-objdump -d  - >%t2
# RUN: FileCheck --input-file=%t2 %s --check-prefixes=CHECK,LONG-NOP

# CHECK: 0000000000000000 <foo>:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            18: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       1b: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       1d: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       20: 74 5d                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            3a: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       3c: 74 41                            je      {{.*}}
# CHECK-NEXT:       3e: 5d                               popq    %rbp
# CHECK-NEXT:       3f: 90                               nop
# CHECK-NEXT:       40: 74 3d                            je      {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            5a: 31 c0                            xorl    %eax, %eax
# CHECK-NEXT:       5c: eb 27                            jmp     {{.*}}
# SHORT-NOP-COUNT-2:  : 90                               nop
# LONG-NOP:         5e: 66 90                            nop
# CHECK-NEXT:       60: eb 23                            jmp     {{.*}}
# CHECK-NEXT:       62: eb 21                            jmp     {{.*}}
# CHECK-COUNT-2:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            74: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK-NEXT:       77: 5d                               popq    %rbp
# CHECK-NEXT:       78: 48 39 c5                         cmpq    %rax, %rbp
# CHECK-NEXT:       7b: 74 02                            je      {{.*}}
# CHECK-NEXT:       7d: eb 06                            jmp     {{.*}}
# CHECK-NEXT:       7f: 8b 45 f4                         movl    -12(%rbp), %eax
# CHECK-NEXT:       82: 89 45 fc                         movl    %eax, -4(%rbp)
# CHECK-COUNT-10:     : 89 b5 50 fb ff ff                movl    %esi, -1200(%rbp)
# CHECK:            c1: eb c2                            jmp     {{.*}}
# CHECK-NEXT:       c3: c3                               retq
