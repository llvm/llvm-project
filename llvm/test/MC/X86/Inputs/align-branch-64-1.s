  .text
  .globl  foo
  .p2align  4
foo:
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  cmp  %rax, %rbp
  xorl %eax, %eax
  cmp  %rax, %rbp
  je  .L_2
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  xorl %eax, %eax
  je  .L_2
  popq  %rbp
  je  .L_2
  .rept 3
  movl  %eax, %fs:0x1
  .endr
  xorl %eax, %eax
  jmp  .L_3
  jmp  .L_3
  jmp  .L_3
  .rept 2
  movl  %eax, %fs:0x1
  .endr
  movl  %eax, -4(%rbp)
  popq  %rbp
  cmp  %rax, %rbp
  je  .L_2
  jmp  .L_3
.L_2:
  movl  -12(%rbp), %eax
  movl  %eax, -4(%rbp)
.L_3:
  .rept 10
  movl  %esi, -1200(%rbp)
  .endr
  jmp  .L_3
  retq

