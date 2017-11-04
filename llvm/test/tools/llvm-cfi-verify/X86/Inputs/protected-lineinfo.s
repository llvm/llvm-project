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
# Compile with (output is in tiny.s.0):
#    clang++ -flto -fsanitize=cfi -fvisibility=hidden -c tiny.cc -o tiny.o -gmlt
#    clang++ tiny.o -o tiny -flto -fuse-ld=gold -Wl,-plugin-opt,save-temps
#    clang++ -fsanitize=cfi -flto -fvisibility=hidden -c tiny.cc -o tiny.o -gmlt
#    llvm-lto2 run @tiny.resolution.txt -o tiny.s -filetype=asm

  .text
  .file "ld-temp.o"
  .p2align  4, 0x90
  .type _Z1av.cfi,@function
_Z1av.cfi:
.Lfunc_begin0:
  .file 1 "tiny.cc"
  .loc  1 1 0
  .cfi_startproc
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
.Ltmp0:
  .loc  1 1 11 prologue_end
  popq  %rbp
  retq
.Ltmp1:
.Lfunc_end0:
  .size _Z1av.cfi, .Lfunc_end0-_Z1av.cfi
  .cfi_endproc

  .p2align  4, 0x90
  .type _Z1bv.cfi,@function
_Z1bv.cfi:
.Lfunc_begin1:
  .loc  1 2 0
  .cfi_startproc
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
.Ltmp2:
  .loc  1 2 11 prologue_end
  popq  %rbp
  retq
.Ltmp3:
.Lfunc_end1:
  .size _Z1bv.cfi, .Lfunc_end1-_Z1bv.cfi
  .cfi_endproc

  .hidden main
  .globl  main
  .p2align  4, 0x90
  .type main,@function
main:
.Lfunc_begin2:
  .loc  1 4 0
  .cfi_startproc
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  subq  $32, %rsp
  movl  $0, -8(%rbp)
  movl  %edi, -4(%rbp)
  movq  %rsi, -24(%rbp)
.Ltmp4:
  .loc  1 6 12 prologue_end
  cmpl  $1, -4(%rbp)
  .loc  1 6 7 is_stmt 0
  jne .LBB2_2
  .loc  1 0 7
  leaq  _Z1av(%rip), %rax
  .loc  1 7 9 is_stmt 1
  movq  %rax, -16(%rbp)
  .loc  1 7 5 is_stmt 0
  jmp .LBB2_3
.LBB2_2:
  .loc  1 0 5
  leaq  _Z1bv(%rip), %rax
  .loc  1 9 9 is_stmt 1
  movq  %rax, -16(%rbp)
.LBB2_3:
  .loc  1 0 9 is_stmt 0
  leaq  .L.cfi.jumptable(%rip), %rcx
  .loc  1 11 3 is_stmt 1
  movq  -16(%rbp), %rax
  movq  %rax, %rdx
  subq  %rcx, %rdx
  movq  %rdx, %rcx
  shrq  $3, %rcx
  shlq  $61, %rdx
  orq %rcx, %rdx
  cmpq  $1, %rdx
  jbe .LBB2_5
  ud2
.LBB2_5:
  callq *%rax
  .loc  1 12 1
  movl  -8(%rbp), %eax
  addq  $32, %rsp
  popq  %rbp
  retq
.Ltmp5:
.Lfunc_end2:
  .size main, .Lfunc_end2-main
  .cfi_endproc

  .p2align  3, 0x90
  .type .L.cfi.jumptable,@function
.L.cfi.jumptable:
.Lfunc_begin3:
  .cfi_startproc
  #APP
  jmp _Z1av.cfi@PLT
  int3
  int3
  int3
  jmp _Z1bv.cfi@PLT
  int3
  int3
  int3

  #NO_APP
.Lfunc_end3:
  .size .L.cfi.jumptable, .Lfunc_end3-.L.cfi.jumptable
  .cfi_endproc

  .section  .debug_str,"MS",@progbits,1
.Linfo_string0:
  .asciz  "clang version 6.0.0 (trunk 316774)"
.Linfo_string1:
  .asciz  "tiny.cc"
.Linfo_string2:
  .asciz  ""
  .section  .debug_abbrev,"",@progbits
  .byte 1
  .byte 17
  .byte 0
  .byte 37
  .byte 14
  .byte 19
  .byte 5
  .byte 3
  .byte 14
  .byte 16
  .byte 23
  .byte 27
  .byte 14
  .byte 17
  .byte 1
  .byte 18
  .byte 6
  .byte 0
  .byte 0
  .byte 0
  .section  .debug_info,"",@progbits
.Lcu_begin0:
  .long 38
  .short  4
  .long .debug_abbrev
  .byte 8
  .byte 1
  .long .Linfo_string0
  .short  4
  .long .Linfo_string1
  .long .Lline_table_start0
  .long .Linfo_string2
  .quad .Lfunc_begin0
  .long .Lfunc_end2-.Lfunc_begin0
  .section  .debug_ranges,"",@progbits
  .section  .debug_macinfo,"",@progbits
.Lcu_macro_begin0:
  .byte 0

  .type _Z1av,@function
_Z1av = .L.cfi.jumptable
  .type _Z1bv,@function
_Z1bv = .L.cfi.jumptable+8
  .ident  "clang version 6.0.0 (trunk 316774)"
  .section  ".note.GNU-stack","",@progbits
  .section  .debug_line,"",@progbits
.Lline_table_start0:

