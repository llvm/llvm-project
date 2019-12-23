# Check NOP padding is disabled before instruction that has variant symbol operand.
# RUN: llvm-mc -filetype=obj -triple i386-unknown-unknown --x86-align-branch-boundary=32 --x86-align-branch=call %s | llvm-objdump -d  - | FileCheck %s

# CHECK: 00000000 foo:
# CHECK-COUNT-5:      : 64 a3 01 00 00 00                movl    %eax, %fs:1
# CHECK:            1e: e8 fc ff ff ff                   calll   {{.*}}
# CHECK-COUNT-4:      : 64 a3 01 00 00 00                movl    %eax, %fs:1
# CHECK:            3b: 55                               pushl   %ebp
# CHECK-NEXT:       3c: ff 91 00 00 00 00                calll   *(%ecx)
# CHECK-COUNT-4:      : 64 a3 01 00 00 00                movl    %eax, %fs:1
# CHECK:            5a: c1 e9 02                         shrl    $2, %ecx
# CHECK-NEXT:       5d: 55                               pushl   %ebp
# CHECK-NEXT:       5e: ff 10                            calll   *(%eax)
# CHECK-COUNT-5:      : 64 a3 01 00 00 00                movl    %eax, %fs:1
# CHECK-NEXT:       7e: ff 20                            jmpl    *(%eax)
  .text
  .globl  foo
  .p2align  4
foo:
  .rept 5
  movl  %eax, %fs:0x1
  .endr
  call    ___tls_get_addr@PLT
  .rept 4
  movl  %eax, %fs:0x1
  .endr
  pushl  %ebp
  call *___tls_get_addr@GOT(%ecx)
  .rept 4
  movl  %eax, %fs:0x1
  .endr
  shrl  $2, %ecx
  pushl %ebp
  call *foo@tlscall(%eax)
  .rept 5
  movl  %eax, %fs:0x1
  .endr
  jmp  *foo@tlscall(%eax)
