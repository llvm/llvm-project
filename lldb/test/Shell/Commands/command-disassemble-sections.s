## Test disassembling of functions which are spread over multiple sections (ELF
## segments are modelled as LLDB sections).


# REQUIRES: x86, lld

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %t/file.s -o %t/file.o
# RUN: ld.lld %t/file.o -o %t/file.out -T %t/file.lds
# RUN: %lldb %t/file.out -o "disassemble --name func1" -o exit | FileCheck %s

# CHECK:      (lldb) disassemble --name func1
# CHECK:      file.out`func1:
# CHECK-NEXT: file.out[0x0] <+0>: int    $0x2a
# CHECK:      file.out`func1:
# CHECK-NEXT: file.out[0x1000] <+4096>: int    $0x2f


#--- file.lds
## Linker script placing the parts of the section into different segments
## (typically one of these would be for the "hot" code).
PHDRS {
  text1 PT_LOAD;
  text2 PT_LOAD;
}
SECTIONS {
  . = 0;
  .text.part1 : { *(.text.part1) } :text1
  .text.part2 : { *(.text.part2) } :text2
}

#--- file.s
## A very simple function consisting of two parts and DWARF describing the
## function.
        .section        .text.part1,"ax",@progbits
        .p2align 12
func1:
        int $42
.Lfunc1_end:

        .section        .text.part2,"ax",@progbits
        .p2align 12
func1.__part.1:
        int $47
.Lfunc1.__part.1_end:



        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   85                              # DW_AT_ranges
        .byte   23                              # DW_FORM_sec_offset
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   0                               # DW_CHILDREN_no
        .byte   85                              # DW_AT_ranges
        .byte   23                              # DW_FORM_sec_offset
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   1                               # Abbrev DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .short  29                              # DW_AT_language
        .quad   0                               # DW_AT_low_pc
        .long   .Ldebug_ranges0                 # DW_AT_ranges
        .byte   2                               # Abbrev DW_TAG_subprogram
        .long   .Ldebug_ranges0                 # DW_AT_ranges
        .asciz  "func1"                         # DW_AT_name
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_rnglists,"",@progbits
        .long   .Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
        .short  5                               # Version
        .byte   8                               # Address size
        .byte   0                               # Segment selector size
        .long   1                               # Offset entry count
.Lrnglists_table_base0:
        .long   .Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
        .byte   6                               # DW_RLE_start_end
        .quad func1
        .quad .Lfunc1_end
        .byte   6                               # DW_RLE_start_end
        .quad func1.__part.1
        .quad .Lfunc1.__part.1_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
