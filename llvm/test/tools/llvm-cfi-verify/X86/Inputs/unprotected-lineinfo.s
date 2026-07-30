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
#    clang++ -gmlt tiny.cc -S -o tiny.s

  .text
  .file "tiny.cc"
  .globl  _Z1av                   # -- Begin function _Z1av
  .p2align  4, 0x90
  .type _Z1av,@function
_Z1av:                                  # @_Z1av
.Lfunc_begin0:
  .file 1 "tiny.cc"
  .loc  1 1 0                   # tiny.cc:1:0
  .cfi_startproc
# %bb.0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
.Ltmp0:
  .loc  1 1 11 prologue_end     # tiny.cc:1:11
  popq  %rbp
  retq
.Ltmp1:
.Lfunc_end0:
  .size _Z1av, .Lfunc_end0-_Z1av
  .cfi_endproc
                                        # -- End function
  .globl  _Z1bv                   # -- Begin function _Z1bv
  .p2align  4, 0x90
  .type _Z1bv,@function
_Z1bv:                                  # @_Z1bv
.Lfunc_begin1:
  .loc  1 2 0                   # tiny.cc:2:0
  .cfi_startproc
# %bb.0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
.Ltmp2:
  .loc  1 2 11 prologue_end     # tiny.cc:2:11
  popq  %rbp
  retq
.Ltmp3:
.Lfunc_end1:
  .size _Z1bv, .Lfunc_end1-_Z1bv
  .cfi_endproc
                                        # -- End function
  .globl  main                    # -- Begin function main
  .p2align  4, 0x90
  .type main,@function
main:                                   # @main
.Lfunc_begin2:
  .loc  1 4 0                   # tiny.cc:4:0
  .cfi_startproc
# %bb.0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  subq  $32, %rsp
  movl  $0, -4(%rbp)
  movl  %edi, -8(%rbp)
  movq  %rsi, -16(%rbp)
.Ltmp4:
  .loc  1 6 12 prologue_end     # tiny.cc:6:12
  cmpl  $1, -8(%rbp)
  .loc  1 6 7 is_stmt 0         # tiny.cc:6:7
  jne .LBB2_2
# %bb.1:
  .loc  1 0 7                   # tiny.cc:0:7
  movabsq $_Z1av, %rax
  .loc  1 7 9 is_stmt 1         # tiny.cc:7:9
  movq  %rax, -24(%rbp)
  .loc  1 7 5 is_stmt 0         # tiny.cc:7:5
  jmp .LBB2_3
.LBB2_2:
  .loc  1 0 5                   # tiny.cc:0:5
  movabsq $_Z1bv, %rax
  .loc  1 9 9 is_stmt 1         # tiny.cc:9:9
  movq  %rax, -24(%rbp)
.LBB2_3:
  .loc  1 11 3                  # tiny.cc:11:3
  callq *-24(%rbp)
  .loc  1 12 1                  # tiny.cc:12:1
  movl  -4(%rbp), %eax
  addq  $32, %rsp
  popq  %rbp
  retq
.Ltmp5:
.Lfunc_end2:
  .size main, .Lfunc_end2-main
  .cfi_endproc
                                        # -- End function
  .section  .debug_str,"MS",@progbits,1
.Linfo_string0:
  .asciz  "clang version 6.0.0 (trunk 316774)" # string offset=0
.Linfo_string1:
  .asciz  "tiny.cc"               # string offset=35
.Linfo_string2:
  .asciz  "/tmp/a/b"              # string offset=43
  .section  .debug_abbrev,"",@progbits
  .byte 1                       # Abbreviation Code
  .byte 17                      # DW_TAG_compile_unit
  .byte 0                       # DW_CHILDREN_no
  .byte 37                      # DW_AT_producer
  .byte 14                      # DW_FORM_strp
  .byte 19                      # DW_AT_language
  .byte 5                       # DW_FORM_data2
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 16                      # DW_AT_stmt_list
  .byte 23                      # DW_FORM_sec_offset
  .byte 27                      # DW_AT_comp_dir
  .byte 14                      # DW_FORM_strp
  .byte 17                      # DW_AT_low_pc
  .byte 1                       # DW_FORM_addr
  .byte 18                      # DW_AT_high_pc
  .byte 6                       # DW_FORM_data4
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 0                       # EOM(3)
  .section  .debug_info,"",@progbits
.Lcu_begin0:
  .long 38                      # Length of Unit
  .short  4                       # DWARF version number
  .long .debug_abbrev           # Offset Into Abbrev. Section
  .byte 8                       # Address Size (in bytes)
  .byte 1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
  .long .Linfo_string0          # DW_AT_producer
  .short  4                       # DW_AT_language
  .long .Linfo_string1          # DW_AT_name
  .long .Lline_table_start0     # DW_AT_stmt_list
  .long .Linfo_string2          # DW_AT_comp_dir
  .quad .Lfunc_begin0           # DW_AT_low_pc
  .long .Lfunc_end2-.Lfunc_begin0 # DW_AT_high_pc
  .section  .debug_ranges,"",@progbits
  .section  .debug_macinfo,"",@progbits
.Lcu_macro_begin0:
  .byte 0                       # End Of Macro List Mark

  .ident  "clang version 6.0.0 (trunk 316774)"
  .section  ".note.GNU-stack","",@progbits
  .section  .debug_line,"",@progbits
.Lline_table_start0:
