# Check no nop is inserted if no branch cross or is against the boundary
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+jmp+indirect+call+ret  %s | llvm-objdump -d  - > %t1
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s | llvm-objdump -d  - > %t2
# RUN: cmp %t1 %t2
# RUN: FileCheck --input-file=%t1 %s

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            18: c1 e9 02                         shrl    $2, %ecx
# CHECK-NEXT:       1b: 89 d1                            movl    %edx, %ecx
# CHECK-NEXT:       1d: 75 fc                            jne     {{.*}}
# CHECK-NEXT:       1f: 55                               pushq   %rbp
# CHECK-NEXT:       20: f6 c2 02                         testb   $2, %dl
# CHECK-NEXT:       23: 75 fa                            jne     {{.*}}
# CHECK-COUNT-2:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            35: c1 e9 02                         shrl    $2, %ecx
# CHECK-NEXT:       38: e8 c3 ff ff ff                   callq   {{.*}}
# CHECK-NEXT:       3d: ff e0                            jmpq    *%rax
# CHECK-NEXT:       3f: 55                               pushq   %rbp
# CHECK-NEXT:       40: c2 63 00                         retq    $99

    .text
    .p2align 4
foo:
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  shrl  $2, %ecx
.L1:
  movl  %edx, %ecx
  jne   .L1
.L2:
  push  %rbp
  testb $2, %dl
  jne    .L2
  .rept 2
  movl  %eax, %fs:0x1
  .endr
  shrl  $2, %ecx
  call  foo
  jmp   *%rax
  push  %rbp
  ret   $99
