 .section  __TEXT,__text,regular,pure_instructions

  .globl  _form_func_ptr
_form_func_ptr:
  leaq  _form_func_ptr(%rip), %rax
  leaq  _other(%rip), %rax
  leaq  _form_func_ptr(%rip), %rax
  nop
  leaq  _form_func_ptr(%rip), %rax
  retq

  .globl  _other
_other:
  leaq  _form_func_ptr(%rip), %rax
  retq

# Return 0 if the pointers formed inside and outside the function are the same.
  .globl _main
_main:
  pushq %rbp
  movq  %rsp, %rbp
  subq  $32, %rsp
  movl  $0, -4(%rbp)
  callq _form_func_ptr
  movq  %rax, -16(%rbp)
  callq _other
  movq  %rax, -24(%rbp)
  movq  -16(%rbp), %rax
  cmpq  -24(%rbp), %rax
  setne  %al
  andb  $1, %al
  movzbl  %al, %eax
  addq  $32, %rsp
  popq  %rbp
  retq