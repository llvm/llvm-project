## This test checks that lldb handles (corrupt?) debug info which has improperly
## nested blocks. The behavior here is not prescriptive. We only want to check
## that we do something "reasonable".


# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s look_me_up1" \
# RUN:   -o "image lookup -v -s look_me_up2" -o exit 2>&1 | FileCheck %s

# CHECK-LABEL: image lookup -v -s look_me_up1
# CHECK: warning: {{.*}} block 0x55 has a range [0x2, 0x4) which is not contained in the parent block 0x44
# CHECK:    Function: id = {0x00000030}, name = "fn", range = [0x0000000000000000-0x0000000000000005)
# CHECK:      Blocks: id = {0x00000030}, range = [0x00000000-0x00000005)
# CHECK-NEXT:         id = {0x00000044}, range = [0x00000001-0x00000003)
# CHECK-NEXT:         id = {0x00000055}, range = [0x00000002-0x00000004)
# CHECK-NEXT: Symbol:

# CHECK-LABEL: image lookup -v -s look_me_up2
# CHECK:    Function: id = {0x00000030}, name = "fn", range = [0x0000000000000000-0x0000000000000005)
# CHECK:      Blocks: id = {0x00000030}, range = [0x00000000-0x00000005)
# CHECK-NEXT: Symbol:

        .text
        .p2align 12
fn:
        nop
.Lblock1_begin:
        nop
.Lblock2_begin:
look_me_up1:
        nop
.Lblock1_end:
look_me_up2:
        nop
.Lblock2_end:
        nop
.Lfn_end:


        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   1                       # DW_FORM_addr
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   1                       # DW_FORM_addr
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   11                      # DW_TAG_lexical_block
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   1                       # DW_FORM_addr
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .quad   fn                      # DW_AT_low_pc
        .quad   .Lfn_end                # DW_AT_high_pc
        .byte   2                       # Abbrev DW_TAG_subprogram
        .quad   fn                      # DW_AT_low_pc
        .quad   .Lfn_end                # DW_AT_high_pc
        .asciz  "fn"                    # DW_AT_name
        .byte   3                       # Abbrev DW_TAG_lexical_block
        .quad   .Lblock1_begin          # DW_AT_low_pc
        .quad   .Lblock1_end            # DW_AT_high_pc
        .byte   3                       # Abbrev DW_TAG_lexical_block
        .quad   .Lblock2_begin          # DW_AT_low_pc
        .quad   .Lblock2_end            # DW_AT_high_pc
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
