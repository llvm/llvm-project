# Check NOP padding is disabled before instruction that has variant symbol operand.
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=jmp+call %s | llvm-objdump -d  - | FileCheck %s

# CHECK: 0000000000000000 foo:
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK-COUNT-2:      : 48 89 e5                         movq    %rsp, %rbp
# CHECK:            1e: e8 00 00 00 00                   callq   {{.*}}
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            3b: 55                               pushq   %rbp
# CHECK-NEXT:       3c: 89 75 f4                         movl    %esi, -12(%rbp)
# CHECK-NEXT:       3f: ff 15 00 00 00 00                callq   *(%rip)
# CHECK-COUNT-3:      : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            5d: ff 15 00 00 00 00                callq   *(%rip)
# CHECK-NEXT-3:       : 64 89 04 25 01 00 00 00          movl    %eax, %fs:1
# CHECK:            7b: ff 25 00 00 00 00                jmpq    *(%rip)

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
  call    __tls_get_addr@PLT
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  pushq  %rbp
  movl  %esi, -12(%rbp)
  call    *__tls_get_addr@GOTPCREL(%rip)
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  call *foo@GOTPCREL(%rip)
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  jmp  *foo@GOTPCREL(%rip)
