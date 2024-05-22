# REQUIRES: native && x86_64-linux

# FIXME: Investigate why broken with MSAN
# UNSUPPORTED: msan

# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t/ELF_x86-64_perf.o %s
# RUN: JITDUMPDIR="%t" llvm-jitlink -perf-support \
# RUN:     %t/ELF_x86-64_perf.o
# RUN: test -f %t/.debug/jit/llvm-IR-jit-*/jit-*.dump

# Test ELF perf support for code load records and unwind info

        .text
        .file   "example.c"
        .section        .text.source,"ax",@progbits
        .globl  source                          # -- Begin function source
        .p2align        4, 0x90
        .type   source,@function
source:                                 # @source
.Lfunc_begin0:
        .file   1 "/app" "example.c"
        .loc    1 1 0                           # example.c:1:0
        .cfi_startproc
# %bb.0:
        .loc    1 2 5 prologue_end              # example.c:2:5
        movl    $1, %eax
        retq
.Ltmp0:
.Lfunc_end0:
        .size   source, .Lfunc_end0-source
        .cfi_endproc
                                        # -- End function
        .section        .text.passthrough,"ax",@progbits
        .globl  passthrough                     # -- Begin function passthrough
        .p2align        4, 0x90
        .type   passthrough,@function
passthrough:                            # @passthrough
.Lfunc_begin1:
        .loc    1 5 0                           # example.c:5:0
        .cfi_startproc
# %bb.0:
        .loc    1 6 5 prologue_end              # example.c:6:5
        movl    $1, %eax
        retq
.Ltmp1:
.Lfunc_end1:
        .size   passthrough, .Lfunc_end1-passthrough
        .cfi_endproc
                                        # -- End function
        .section        .text.main,"ax",@progbits
        .globl  main                            # -- Begin function main
        .p2align        4, 0x90
        .type   main,@function
main:                                   # @main
.Lfunc_begin2:
        .loc    1 9 0                           # example.c:9:0
        .cfi_startproc
# %bb.0:
        .loc    1 10 5 prologue_end             # example.c:10:5
        xorl    %eax, %eax
        retq
.Ltmp2:
.Lfunc_end2:
        .size   main, .Lfunc_end2-main
        .cfi_endproc
                                        # -- End function
        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   14                              # DW_FORM_strp
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   16                              # DW_AT_stmt_list
        .byte   23                              # DW_FORM_sec_offset
        .byte   27                              # DW_AT_comp_dir
        .byte   14                              # DW_FORM_strp
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   85                              # DW_AT_ranges
        .byte   23                              # DW_FORM_sec_offset
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   0                               # DW_CHILDREN_no
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
        .ascii  "\227B"                         # DW_AT_GNU_all_call_sites
        .byte   25                              # DW_FORM_flag_present
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                               # DWARF version number
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .byte   1                               # Abbrev [1] 0xb:0x72 DW_TAG_compile_unit
        .long   .Linfo_string0                  # DW_AT_producer
        .short  12                              # DW_AT_language
        .long   .Linfo_string1                  # DW_AT_name
        .long   .Lline_table_start0             # DW_AT_stmt_list
        .long   .Linfo_string2                  # DW_AT_comp_dir
        .quad   0                               # DW_AT_low_pc
        .long   .Ldebug_ranges0                 # DW_AT_ranges
        .byte   2                               # Abbrev [2] 0x2a:0x19 DW_TAG_subprogram
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   87
                                        # DW_AT_GNU_all_call_sites
        .long   .Linfo_string3                  # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   1                               # DW_AT_decl_line
        .long   117                             # DW_AT_type
                                        # DW_AT_external
        .byte   2                               # Abbrev [2] 0x43:0x19 DW_TAG_subprogram
        .quad   .Lfunc_begin1                   # DW_AT_low_pc
        .long   .Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   87
                                        # DW_AT_GNU_all_call_sites
        .long   .Linfo_string5                  # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   5                               # DW_AT_decl_line
        .long   117                             # DW_AT_type
                                        # DW_AT_external
        .byte   2                               # Abbrev [2] 0x5c:0x19 DW_TAG_subprogram
        .quad   .Lfunc_begin2                   # DW_AT_low_pc
        .long   .Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   87
                                        # DW_AT_GNU_all_call_sites
        .long   .Linfo_string6                  # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   9                               # DW_AT_decl_line
        .long   117                             # DW_AT_type
                                        # DW_AT_external
        .byte   3                               # Abbrev [3] 0x75:0x7 DW_TAG_base_type
        .long   .Linfo_string4                  # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
        .section        .debug_ranges,"",@progbits
.Ldebug_ranges0:
        .quad   .Lfunc_begin0
        .quad   .Lfunc_end0
        .quad   .Lfunc_begin1
        .quad   .Lfunc_end1
        .quad   .Lfunc_begin2
        .quad   .Lfunc_end2
        .quad   0
        .quad   0
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 4ba6a9c9f65bbc8bd06e3652cb20fd4dfc846137)" # string offset=0
.Linfo_string1:
        .asciz  "/app/example.c"                # string offset=105
.Linfo_string2:
        .asciz  "/app"                          # string offset=120
.Linfo_string3:
        .asciz  "source"                        # string offset=125
.Linfo_string4:
        .asciz  "int"                           # string offset=132
.Linfo_string5:
        .asciz  "passthrough"                   # string offset=136
.Linfo_string6:
        .asciz  "main"                          # string offset=148
        .ident  "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 4ba6a9c9f65bbc8bd06e3652cb20fd4dfc846137)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .section        .debug_line,"",@progbits
.Lline_table_start0:
