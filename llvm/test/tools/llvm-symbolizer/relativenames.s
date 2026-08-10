# REQUIRES: x86-registered-target

## Test that the relative names option to llvm-symbolizer works properly.
## See llvm/docs/CommandGuide/llvm-symbolizer.rst for commands
## that would produce this test case

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o

# RUN: llvm-symbolizer 0 --relativenames --obj=%t.o \
# RUN:    | FileCheck %s -DDIR=%p --check-prefix=RELATIVENAMES

## A basic correctness check for default.
# RUN: llvm-symbolizer 0 --obj=%t.o \
# RUN:    | FileCheck %s -DDIR=%p --check-prefix=ABSOLUTENAMES

## Ensure last option wins.
# RUN: llvm-symbolizer 0 --basenames --relativenames --obj=%t.o \
# RUN:    | FileCheck %s -DDIR=%p --check-prefix=RELATIVENAMES
# RUN: llvm-symbolizer 0 --relativenames --basenames --obj=%t.o \
# RUN:    | FileCheck %s --check-prefix=BASENAMES

# ABSOLUTENAMES: {{\\|/}}tmp{{\\|/}}foo{{\\|/}}relativenames.s:4
# RELATIVENAMES: {{^}}foo{{\\|/}}relativenames.s:4
# BASENAMES: {{^}}relativenames.s:4

##  Provide just enough info in debug info for the symbolizer
##  to find the line table.
.section .debug_abbrev,"",@progbits
    .byte 0x01  ## Abbreviation Code
    .byte 0x11  ## DW_TAG_compile_unit
    .byte 0x01  ## DW_CHILDREN_yes
    .byte 0x10  ## DW_AT_stmt_list
    .byte 0x17  ## DW_FORM_sec_offset
    .byte 0x11  ## DW_AT_low_pc
    .byte 0x01  ## DW_FORM_addr
    .byte 0x12  ## DW_AT_high_pc
    .byte 0x01  ## DW_FORM_addr
    .byte 0x1b  ## DW_AT_comp_dir
    .byte 0x08  ## DW_FORM_string
    .byte 0x03  ## DW_AT_name
    .byte 0x08  ## DW_FORM_string
    .byte 0x00  ## EOM(1)
    .byte 0x00  ## EOM(2)
    .byte 0x02  ## Abbreviation Code
    .byte 0x2e  ## DW_TAG_subprogram
    .byte 0x0   ## DW_CHILDREN_no
    .byte 0x11  ## DW_AT_low_pc
    .byte 0x01  ## DW_FORM_addr
    .byte 0x12  ## DW_AT_high_pc
    .byte 0x01  ## DW_FORM_addr
    .byte 0x00  ## EOM(1)
    .byte 0x00  ## EOM(2)
    .byte 0x00  ## EOM(3)

.section .debug_info,"",@progbits
    .long   .Ldebug_info_end-.Ldebug_info_start ## length
.Ldebug_info_start:
    .short 0x05  ## version
    .byte  0x01  ## DW_TAG_compile_unit
    .byte  0x08  ## address size
    .long  .debug_abbrev   ## offset into abbrev section
    .byte  0x01  ## cu abbrev code
    .long  .debug_line  ## DW_AT_stmt_list
    .quad  0x00  ## DW_AT_low_pc
    .quad  0x01  ## DW_AT_high_pc
    .asciz "/tmp"  ## DW_AT_comp_dir
    .asciz "foo/relativenames.s" ## DW_AT_name
    .byte  0x02  ## subprog abbrev code
    .quad  0x00  ## DW_AT_low_pc
    .quad  0x01  ## DW_AT_high_pc
.Ldebug_info_end:


.section .debug_line,"",@progbits
    .long .Lunit_end - .Lunit_begin ## unit_length
.Lunit_begin:
    .short 0x05  ## version
    .byte  0x08  ## address_size
    .byte  0x00  ## segment_selector_size
    .long .Lheader_end - .Lheader_begin ## header_length
.Lheader_begin:
    .byte  0x01  ## minimum_instruction_length
    .byte  0x01  ## maximum_operations_per_instruction
    .byte  0x01  ## default_is_stmt
    .byte  0xfb  ## line_base
    .byte  0x0e  ## line_range
    .byte  0x0d  ## opcode_base and lengths
    .byte  0x00, 0x01, 0x01, 0x01, 0x01, 0x00
    .byte  0x00, 0x00, 0x01, 0x00, 0x00, 0x01
    .byte  0x01  ## directory entry format count
    .byte  0x01, 0x8 ## DW_LNCT_path, DW_FORM_string
    .byte  0x01  ## directories count
    .asciz "/tmp" ## directory entry 0
    .byte  0x02  ## file_name_entry_format_count
    .byte  0x02, 0x0B  ## DW_LNCT_directory_index, DW_FORM_data1
    .byte  0x01, 0x08  ## DW_LNCT_path, DW_FORM_string
    .byte  0x01  ## filname count
    .byte  0x00  ## directory index
    .asciz "foo/relativenames.s"
.Lheader_end:
    .byte 0x04, 0x00 ## set file to zero
    ## set address to 0x0
    .byte 0x00, 0x09, 0x02, 0x00, 0x00, 0x00, 0x00
    .byte 0x00, 0x00, 0x00, 0x00
    .byte 0x15  ## Advance Address by 0 and line by 3
    .byte 0x02, 0x01  ## Advance PC by 1
    .byte 0x0, 0x1, 0x1  ## DW_LNE_end_sequence
.Lunit_end:
