  .text
  .globl  foo
  .p2align  4
foo:
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  .rept 2
  movl  %esi, -12(%rbp)
  .endr
  jmp  *%rax
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  movl  %esi, -12(%rbp)
  pushq  %rbp
  call *%rax
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  pushq  %rbp
  call  foo
  .rept 4
  movl  %eax, %fs:0x1
  .endr
  call  *foo

