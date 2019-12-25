# Check only indirect jumps are aligned with option --x86-align-branch-boundary=32 --x86-align-branch=indirect
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=indirect %p/Inputs/align-branch-64-2.s | llvm-objdump -d  - | FileCheck %s

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-COUNT-2:      : 89 75 f4                         movl    %esi, -12(%rbp)
# CHECK-COUNT-2:      : 90                               nop
# CHECK:            20: ff e0                            jmpq    *%rax
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            3a: 89 75 f4                         movl    %esi, -12(%rbp)
# CHECK-NEXT:       3d: 55                               pushq   %rbp
# CHECK-NEXT:       3e: ff d0                            callq   *%rax
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       58: 55                               pushq   %rbp
# CHECK-NEXT:       59: e8 a2 ff ff ff                   callq   {{.*}}
# CHECK-COUNT-4:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            7e: ff 14 25 00 00 00 00             callq   *0
