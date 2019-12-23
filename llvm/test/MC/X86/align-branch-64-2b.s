# Check only calls are aligned with option --x86-align-branch-boundary=32 --x86-align-branch=call
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=call %p/Inputs/align-branch-64-2.s | llvm-objdump -d  - | FileCheck %s

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-COUNT-2:      : 89 75 f4                         movl    %esi, -12(%rbp)
# CHECK:            1e: ff e0                            jmpq    *%rax
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            38: 89 75 f4                         movl    %esi, -12(%rbp)
# CHECK-NEXT:       3b: 55                               pushq    %rbp
# CHECK-NEXT:       3c: ff d0                            callq    *%rax
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl     %eax, %fs:1
# CHECK:            56: 55                               pushq    %rbp
# CHECK-NEXT:       57: e8 a4 ff ff ff                   callq    {{.*}}
# CHECK-COUNT-4:      : 64 89 04 25 01 00 00 00          movl     %eax, %fs:1
# CHECK-COUNT-4:      : 90                               nop
# CHECK:            80: ff 14 25 00 00 00 00             callq    *0
