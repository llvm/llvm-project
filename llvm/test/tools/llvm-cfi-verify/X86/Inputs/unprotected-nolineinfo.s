# Source (tiny.cc):
#   void a() {}
#   void b() {}
#   int main(int argc, char** argv) {
#     void(*ptr)();
#     if (argc == 1)
#       ptr = &a;
#     else
#       ptr = &b;
#     ptr();
#   }
# Compile with:
#    clang++ tiny.cc -S -o tiny.s

  .text
  .file "tiny.cc"
  .globl  _Z1av                   # -- Begin function _Z1av
  .p2align  4, 0x90
  .type _Z1av,@function
_Z1av:                                  # @_Z1av
  .cfi_startproc
# BB#0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  popq  %rbp
  retq
.Lfunc_end0:
  .size _Z1av, .Lfunc_end0-_Z1av
  .cfi_endproc
                                        # -- End function
  .globl  _Z1bv                   # -- Begin function _Z1bv
  .p2align  4, 0x90
  .type _Z1bv,@function
_Z1bv:                                  # @_Z1bv
  .cfi_startproc
# BB#0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  popq  %rbp
  retq
.Lfunc_end1:
  .size _Z1bv, .Lfunc_end1-_Z1bv
  .cfi_endproc
                                        # -- End function
  .globl  main                    # -- Begin function main
  .p2align  4, 0x90
  .type main,@function
main:                                   # @main
  .cfi_startproc
# BB#0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  subq  $32, %rsp
  movl  $0, -4(%rbp)
  movl  %edi, -8(%rbp)
  movq  %rsi, -16(%rbp)
  cmpl  $1, -8(%rbp)
  jne .LBB2_2
# BB#1:
  movabsq $_Z1av, %rax
  movq  %rax, -24(%rbp)
  jmp .LBB2_3
.LBB2_2:
  movabsq $_Z1bv, %rax
  movq  %rax, -24(%rbp)
.LBB2_3:
  callq *-24(%rbp)
  movl  -4(%rbp), %eax
  addq  $32, %rsp
  popq  %rbp
  retq
.Lfunc_end2:
  .size main, .Lfunc_end2-main
  .cfi_endproc
                                        # -- End function

  .ident  "clang version 6.0.0 (trunk 316774)"
  .section  ".note.GNU-stack","",@progbits
