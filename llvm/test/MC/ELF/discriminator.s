# RUN: llvm-mc -triple i386-unknown-unknown %s -filetype=obj -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s
# RUN: llvm-dwarfdump -debug-line %t.o | FileCheck %s -check-prefix=DWARF-DUMP

    .file 1 "foo.c"
    .text
    .globl foo
    .type foo, @function
    .align 4
foo:
    .loc 1 2 discriminator 1
    ret
    .size foo, .-foo

        .section        .debug_info,"",@progbits
.L.debug_info_begin0:
        .long   34                      # Length of Unit
        .short  4                       # DWARF version number
        .long   .L.debug_abbrev_begin   # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x1b DW_TAG_compile_unit
        .long   info_string0            # DW_AT_producer
        .short  12                      # DW_AT_language
        .long   info_string1            # DW_AT_name
        .quad   0                       # DW_AT_low_pc
        .long   0                       # DW_AT_stmt_list
        .long   info_string2            # DW_AT_comp_dir
                                        # DW_AT_APPLE_optimized
        .section        .debug_abbrev,"",@progbits
.L.debug_abbrev_begin:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   37                      # DW_AT_producer
        .byte   14                      # DW_FORM_strp
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   27                      # DW_AT_comp_dir
        .byte   14                      # DW_FORM_strp
        .ascii  "\341\177"              # DW_AT_APPLE_optimized
        .byte   25                      # DW_FORM_flag_present
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.L.debug_abbrev_end:


# CHECK:      Relocations [
# CHECK:        Section ({{[^ ]+}}) .rel.debug_line {
# CHECK-NEXT:     0x2D R_386_32 .text 0x0
# CHECK-NEXT:   }

# DWARF-DUMP: Address            Line   Column File   ISA Discriminator Flags
# DWARF-DUMP: ------------------ ------ ------ ------ --- ------------- -------------
# DWARF-DUMP: 0x0001021300000000     1      0      1   0             1  is_stmt
