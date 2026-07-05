# RUN: llvm-mc -triple x86_64-unknown-linux -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump %t.o --debug-loclists | FileCheck %s

# CHECK: DW_LLE_offset_pair     (0x0000000000000000, 0x0000000000000000): <empty>

.Lfunc_begin0:
.Ltmp1:
.section .debug_loclists, "",@progbits
        .long   .Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
        .short  5                               # Version
        .byte   8                               # Address size
        .byte   0                               # Segment selector size
        .long   1                               # Offset entry count
.Lloclists_table_base0:
        .long   .Ldebug_loc0-.Lloclists_table_base0
.Ldebug_loc0:
        .byte   4                               # DW_LLE_offset_pair
        .uleb128 .Lfunc_begin0-.Lfunc_begin0    #   starting offset
        .uleb128 .Ltmp1-.Lfunc_begin0           #   ending offset
        .byte   0                               ### empty
        .byte   0                               # DW_LLE_end_of_list
.Ldebug_list_header_end0:
