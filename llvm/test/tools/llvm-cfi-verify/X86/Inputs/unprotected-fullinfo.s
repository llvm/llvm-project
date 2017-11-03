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
#    clang++ -g tiny.cc -S -o tiny.s

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
# BB#0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
.Ltmp0:
  .loc  1 1 11 prologue_end     # tiny.cc:1:11
  popq  %rbp
  .cfi_def_cfa %rsp, 8
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
# BB#0:
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
.Ltmp2:
  .loc  1 2 11 prologue_end     # tiny.cc:2:11
  popq  %rbp
  .cfi_def_cfa %rsp, 8
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
.Ltmp4:
  .loc  1 6 12 prologue_end     # tiny.cc:6:12
  cmpl  $1, -8(%rbp)
.Ltmp5:
  .loc  1 6 7 is_stmt 0         # tiny.cc:6:7
  jne .LBB2_2
# BB#1:
  .loc  1 0 7                   # tiny.cc:0:7
  movabsq $_Z1av, %rax
.Ltmp6:
  .loc  1 7 9 is_stmt 1         # tiny.cc:7:9
  movq  %rax, -24(%rbp)
  .loc  1 7 5 is_stmt 0         # tiny.cc:7:5
  jmp .LBB2_3
.LBB2_2:
  .loc  1 0 5                   # tiny.cc:0:5
  movabsq $_Z1bv, %rax
  .loc  1 9 9 is_stmt 1         # tiny.cc:9:9
  movq  %rax, -24(%rbp)
.Ltmp7:
.LBB2_3:
  .loc  1 11 3                  # tiny.cc:11:3
  callq *-24(%rbp)
  .loc  1 12 1                  # tiny.cc:12:1
  movl  -4(%rbp), %eax
  addq  $32, %rsp
  popq  %rbp
  .cfi_def_cfa %rsp, 8
  retq
.Ltmp8:
.Lfunc_end2:
  .size main, .Lfunc_end2-main
  .cfi_endproc
                                        # -- End function
  .section  .debug_str,"MS",@progbits,1
.Linfo_string0:
  .asciz  "clang version 6.0.0 (trunk 317104)" # string offset=0
.Linfo_string1:
  .asciz  "tiny.cc"               # string offset=35
.Linfo_string2:
  .asciz  "/tmp/a/b"              # string offset=43
.Linfo_string3:
  .asciz  "_Z1av"                 # string offset=52
.Linfo_string4:
  .asciz  "a"                     # string offset=58
.Linfo_string5:
  .asciz  "_Z1bv"                 # string offset=60
.Linfo_string6:
  .asciz  "b"                     # string offset=66
.Linfo_string7:
  .asciz  "main"                  # string offset=68
.Linfo_string8:
  .asciz  "int"                   # string offset=73
.Linfo_string9:
  .asciz  "argc"                  # string offset=77
.Linfo_string10:
  .asciz  "argv"                  # string offset=82
.Linfo_string11:
  .asciz  "char"                  # string offset=87
.Linfo_string12:
  .asciz  "ptr"                   # string offset=92
  .section  .debug_abbrev,"",@progbits
  .byte 1                       # Abbreviation Code
  .byte 17                      # DW_TAG_compile_unit
  .byte 1                       # DW_CHILDREN_yes
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
  .ascii  "\264B"                 # DW_AT_GNU_pubnames
  .byte 25                      # DW_FORM_flag_present
  .byte 17                      # DW_AT_low_pc
  .byte 1                       # DW_FORM_addr
  .byte 18                      # DW_AT_high_pc
  .byte 6                       # DW_FORM_data4
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 2                       # Abbreviation Code
  .byte 46                      # DW_TAG_subprogram
  .byte 0                       # DW_CHILDREN_no
  .byte 17                      # DW_AT_low_pc
  .byte 1                       # DW_FORM_addr
  .byte 18                      # DW_AT_high_pc
  .byte 6                       # DW_FORM_data4
  .byte 64                      # DW_AT_frame_base
  .byte 24                      # DW_FORM_exprloc
  .byte 110                     # DW_AT_linkage_name
  .byte 14                      # DW_FORM_strp
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 58                      # DW_AT_decl_file
  .byte 11                      # DW_FORM_data1
  .byte 59                      # DW_AT_decl_line
  .byte 11                      # DW_FORM_data1
  .byte 63                      # DW_AT_external
  .byte 25                      # DW_FORM_flag_present
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 3                       # Abbreviation Code
  .byte 46                      # DW_TAG_subprogram
  .byte 1                       # DW_CHILDREN_yes
  .byte 17                      # DW_AT_low_pc
  .byte 1                       # DW_FORM_addr
  .byte 18                      # DW_AT_high_pc
  .byte 6                       # DW_FORM_data4
  .byte 64                      # DW_AT_frame_base
  .byte 24                      # DW_FORM_exprloc
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 58                      # DW_AT_decl_file
  .byte 11                      # DW_FORM_data1
  .byte 59                      # DW_AT_decl_line
  .byte 11                      # DW_FORM_data1
  .byte 73                      # DW_AT_type
  .byte 19                      # DW_FORM_ref4
  .byte 63                      # DW_AT_external
  .byte 25                      # DW_FORM_flag_present
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 4                       # Abbreviation Code
  .byte 5                       # DW_TAG_formal_parameter
  .byte 0                       # DW_CHILDREN_no
  .byte 2                       # DW_AT_location
  .byte 24                      # DW_FORM_exprloc
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 58                      # DW_AT_decl_file
  .byte 11                      # DW_FORM_data1
  .byte 59                      # DW_AT_decl_line
  .byte 11                      # DW_FORM_data1
  .byte 73                      # DW_AT_type
  .byte 19                      # DW_FORM_ref4
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 5                       # Abbreviation Code
  .byte 52                      # DW_TAG_variable
  .byte 0                       # DW_CHILDREN_no
  .byte 2                       # DW_AT_location
  .byte 24                      # DW_FORM_exprloc
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 58                      # DW_AT_decl_file
  .byte 11                      # DW_FORM_data1
  .byte 59                      # DW_AT_decl_line
  .byte 11                      # DW_FORM_data1
  .byte 73                      # DW_AT_type
  .byte 19                      # DW_FORM_ref4
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 6                       # Abbreviation Code
  .byte 36                      # DW_TAG_base_type
  .byte 0                       # DW_CHILDREN_no
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 62                      # DW_AT_encoding
  .byte 11                      # DW_FORM_data1
  .byte 11                      # DW_AT_byte_size
  .byte 11                      # DW_FORM_data1
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 7                       # Abbreviation Code
  .byte 15                      # DW_TAG_pointer_type
  .byte 0                       # DW_CHILDREN_no
  .byte 73                      # DW_AT_type
  .byte 19                      # DW_FORM_ref4
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 8                       # Abbreviation Code
  .byte 21                      # DW_TAG_subroutine_type
  .byte 0                       # DW_CHILDREN_no
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 0                       # EOM(3)
  .section  .debug_info,"",@progbits
.Lcu_begin0:
  .long 187                     # Length of Unit
  .short  4                       # DWARF version number
  .long .debug_abbrev           # Offset Into Abbrev. Section
  .byte 8                       # Address Size (in bytes)
  .byte 1                       # Abbrev [1] 0xb:0xb4 DW_TAG_compile_unit
  .long .Linfo_string0          # DW_AT_producer
  .short  4                       # DW_AT_language
  .long .Linfo_string1          # DW_AT_name
  .long .Lline_table_start0     # DW_AT_stmt_list
  .long .Linfo_string2          # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
  .quad .Lfunc_begin0           # DW_AT_low_pc
  .long .Lfunc_end2-.Lfunc_begin0 # DW_AT_high_pc
  .byte 2                       # Abbrev [2] 0x2a:0x19 DW_TAG_subprogram
  .quad .Lfunc_begin0           # DW_AT_low_pc
  .long .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
  .byte 1                       # DW_AT_frame_base
  .byte 86
  .long .Linfo_string3          # DW_AT_linkage_name
  .long .Linfo_string4          # DW_AT_name
  .byte 1                       # DW_AT_decl_file
  .byte 1                       # DW_AT_decl_line
                                        # DW_AT_external
  .byte 2                       # Abbrev [2] 0x43:0x19 DW_TAG_subprogram
  .quad .Lfunc_begin1           # DW_AT_low_pc
  .long .Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
  .byte 1                       # DW_AT_frame_base
  .byte 86
  .long .Linfo_string5          # DW_AT_linkage_name
  .long .Linfo_string6          # DW_AT_name
  .byte 1                       # DW_AT_decl_file
  .byte 2                       # DW_AT_decl_line
                                        # DW_AT_external
  .byte 3                       # Abbrev [3] 0x5c:0x44 DW_TAG_subprogram
  .quad .Lfunc_begin2           # DW_AT_low_pc
  .long .Lfunc_end2-.Lfunc_begin2 # DW_AT_high_pc
  .byte 1                       # DW_AT_frame_base
  .byte 86
  .long .Linfo_string7          # DW_AT_name
  .byte 1                       # DW_AT_decl_file
  .byte 4                       # DW_AT_decl_line
  .long 160                     # DW_AT_type
                                        # DW_AT_external
  .byte 4                       # Abbrev [4] 0x75:0xe DW_TAG_formal_parameter
  .byte 2                       # DW_AT_location
  .byte 145
  .byte 120
  .long .Linfo_string9          # DW_AT_name
  .byte 1                       # DW_AT_decl_file
  .byte 4                       # DW_AT_decl_line
  .long 160                     # DW_AT_type
  .byte 4                       # Abbrev [4] 0x83:0xe DW_TAG_formal_parameter
  .byte 2                       # DW_AT_location
  .byte 145
  .byte 112
  .long .Linfo_string10         # DW_AT_name
  .byte 1                       # DW_AT_decl_file
  .byte 4                       # DW_AT_decl_line
  .long 167                     # DW_AT_type
  .byte 5                       # Abbrev [5] 0x91:0xe DW_TAG_variable
  .byte 2                       # DW_AT_location
  .byte 145
  .byte 104
  .long .Linfo_string12         # DW_AT_name
  .byte 1                       # DW_AT_decl_file
  .byte 5                       # DW_AT_decl_line
  .long 184                     # DW_AT_type
  .byte 0                       # End Of Children Mark
  .byte 6                       # Abbrev [6] 0xa0:0x7 DW_TAG_base_type
  .long .Linfo_string8          # DW_AT_name
  .byte 5                       # DW_AT_encoding
  .byte 4                       # DW_AT_byte_size
  .byte 7                       # Abbrev [7] 0xa7:0x5 DW_TAG_pointer_type
  .long 172                     # DW_AT_type
  .byte 7                       # Abbrev [7] 0xac:0x5 DW_TAG_pointer_type
  .long 177                     # DW_AT_type
  .byte 6                       # Abbrev [6] 0xb1:0x7 DW_TAG_base_type
  .long .Linfo_string11         # DW_AT_name
  .byte 6                       # DW_AT_encoding
  .byte 1                       # DW_AT_byte_size
  .byte 7                       # Abbrev [7] 0xb8:0x5 DW_TAG_pointer_type
  .long 189                     # DW_AT_type
  .byte 8                       # Abbrev [8] 0xbd:0x1 DW_TAG_subroutine_type
  .byte 0                       # End Of Children Mark
  .section  .debug_ranges,"",@progbits
  .section  .debug_macinfo,"",@progbits
.Lcu_macro_begin0:
  .byte 0                       # End Of Macro List Mark
  .section  .debug_pubnames,"",@progbits
  .long .LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
  .short  2                       # DWARF Version
  .long .Lcu_begin0             # Offset of Compilation Unit Info
  .long 191                     # Compilation Unit Length
  .long 42                      # DIE offset
  .asciz  "a"                     # External Name
  .long 67                      # DIE offset
  .asciz  "b"                     # External Name
  .long 92                      # DIE offset
  .asciz  "main"                  # External Name
  .long 0                       # End Mark
.LpubNames_end0:
  .section  .debug_pubtypes,"",@progbits
  .long .LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
  .short  2                       # DWARF Version
  .long .Lcu_begin0             # Offset of Compilation Unit Info
  .long 191                     # Compilation Unit Length
  .long 160                     # DIE offset
  .asciz  "int"                   # External Name
  .long 177                     # DIE offset
  .asciz  "char"                  # External Name
  .long 0                       # End Mark
.LpubTypes_end0:

  .ident  "clang version 6.0.0 (trunk 317104)"
  .section  ".note.GNU-stack","",@progbits
  .section  .debug_line,"",@progbits
.Lline_table_start0:
