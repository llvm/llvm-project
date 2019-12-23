# Check option --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+jmp+indirect+call+ret can cowork with option --mc-relax-all
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=fused+jcc+jmp+indirect+call+ret --mc-relax-all %s | llvm-objdump -d  - > %t1
# RUN: FileCheck --input-file=%t1 %s

# CHECK: 0000000000000000 foo:
# CHECK-NEXT:        0: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:        8: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       10: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       18: c1 e9 02                         shrl    $2, %ecx
# CHECK-NEXT:       1b: 89 d1                            movl    %edx, %ecx
# CHECK-NEXT:       1d: 90                               nop
# CHECK-NEXT:       1e: 90                               nop
# CHECK-NEXT:       1f: 90                               nop
# CHECK-NEXT:       20: 0f 85 f5 ff ff ff                jne     {{.*}}
# CHECK-NEXT:       26: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       2e: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       36: f6 c2 02                         testb   $2, %dl
# CHECK-NEXT:       39: 0f 85 e7 ff ff ff                jne     {{.*}}
# CHECK-NEXT:       3f: 90                               nop
# CHECK-NEXT:       40: e9 d6 ff ff ff                   jmp     {{.*}}
# CHECK-NEXT:       45: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       4d: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       55: 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-NEXT:       5d: 90                               nop
# CHECK-NEXT:       5e: 90                               nop
# CHECK-NEXT:       5f: 90                               nop
# CHECK-NEXT:       60: e8 9b ff ff ff                   callq   {{.*}}
# CHECK-NEXT:       65: e9 bc ff ff ff                   jmp     {{.*}}
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
  .rept 2
  movl  %eax, %fs:0x1
  .endr
  testb $2, %dl
  jne   .L2
  jmp   .L1
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  call  foo
  jmp   .L2
