## This test checks that looking up a zero hash in the .debug_cu_index hash
## table works correctly when there's no CU with signature = 0.
##
## LLVM used to check just the signature bits to decide if the hash lookup ended
## at a match or at an empty slot. This is wrong when signature = 0 because
## empty slots have all zeros in the signature field too, and LLVM would return
## the empty slot as a valid result.

# REQUIRES: x86-registered-target

# RUN: llvm-mc --filetype=obj --triple x86_64 %s -o %t --defsym MAIN=0
# RUN: llvm-mc --filetype=obj --triple x86_64 %s -o %t.dwp
# RUN: llvm-symbolizer --obj=%t --dwp=%t.dwp 0x0 | FileCheck %s

## This expected output is very uninteresting, but it's better than a crash.
# CHECK: ??:0:0

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .ascii  "\260B"                 # DW_AT_GNU_dwo_name
        .byte   8                       # DW_FORM_string
        .ascii  "\261B"                 # DW_AT_GNU_dwo_id
        .byte   7                       # DW_FORM_data8
        .ascii  "\263B"                 # DW_AT_GNU_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   85                      # DW_AT_ranges
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

## Create two CUs, with dwo_ids 0 and 1 respectively.
.ifdef MAIN
.irpc I,01
        .data
A\I:
        .long \I

        .text
F\I:
        nop

        .section        .debug_info,"",@progbits
.Lcu_begin\I:
        .long   .Ldebug_info_end\I-.Ldebug_info_start\I # Length of Unit
.Ldebug_info_start\I:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x25 DW_TAG_compile_unit
        .asciz  "A.dwo"                 # DW_AT_GNU_dwo_name
        .quad   \I                      # DW_AT_GNU_dwo_id
        .long   .debug_addr             # DW_AT_GNU_addr_base
        .long   .Lranges\I              # DW_AT_ranges
.Ldebug_info_end\I:

        .section        .debug_addr,"",@progbits
        .quad   A\I
        .quad   F\I

        .section        .debug_ranges,"",@progbits
.Lranges\I:
        .quad   F\I
        .quad   F\I+1
        .quad   0
        .quad   0
.endr
.else
## Deliberately omit compile unit 0 in the DWP. We want to check the case where
## a signature = 0 matches an empty hash slot in .debug_cu_index and the index
## in the parallel table has to be checked.
        .section        .debug_abbrev.dwo,"e",@progbits
.Labbrev1:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.Labbrev_end1:

        .section        .debug_info.dwo,"e",@progbits
.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  4                       # DWARF version number
        .long   0                       # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   '1', '.', 'c', 0        # DW_AT_name
.Ldebug_info_end1:

        .section        .debug_cu_index,"",@progbits
        .long   2                       # DWARF version number
        .long   2                       # Section count
        .long   1                       # Unit count
        .long   8                       # Slot count

        .quad   1, 0, 0, 0, 0, 0, 0, 0  # Hash table
        .long   1, 0, 0, 0, 0, 0, 0, 0  # Index table

        .long   1                       # DW_SECT_INFO
        .long   3                       # DW_SECT_ABBREV

        .long .Lcu_begin1-.debug_info.dwo
        .long .Labbrev1-.debug_abbrev.dwo

        .long .Ldebug_info_end1-.Lcu_begin1
        .long .Labbrev_end1-.Labbrev1

.endif
