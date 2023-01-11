# This test checks the support for writing macro sections and their index (v5).

# RUN: llvm-mc -triple x86_64-unknown-linux --filetype=obj --split-dwarf-file=%t.dwo -dwarf-version=5 %s -o %t.o
# RUN: llvm-dwp %t.dwo -o %t.dwp 2>&1
# RUN: llvm-dwarfdump -debug-macro -debug-cu-index %t.dwp | FileCheck -check-prefix=CHECK %s
# RUN: llvm-dwarfdump -debug-macro -debug-cu-index -manaully-generate-unit-index %t.dwp | FileCheck -check-prefix=CHECK2 %s

# CHECK-DAG: .debug_macro.dwo contents:
# CHECK: macro header: version = 0x0005, flags = 0x00, format = DWARF32
# CHECK-NEXT: DW_MACRO_start_file - lineno: 0 filenum: 0
# CHECK-NEXT: DW_MACRO_define_strx - lineno: 1 macro: x 5
# CHECK-NEXT: DW_MACRO_end_file

# CHECK-DAG: .debug_cu_index contents:
# CHECK-NEXT: version = 5, units = 1, slots = 2
# CHECK: Index Signature          INFO                                      ABBREV                   STR_OFFSETS              MACRO
# CHECK:     1 0x0000000000000000 [0x0000000000000000, 0x0000000000000019) [0x00000000, 0x00000008) [0x00000000, 0x0000000c) [0x00000000, 0x0000000b)

# CHECK2: Index Signature          INFO                                     ABBREV                   STR_OFFSETS              MACRO
# CHECK2:     1 0x0000000000000000 [0x0000000000000000, 0x0000000000000019) [0x00000000, 0x00000008) [0x00000000, 0x0000000c) [0x00000000, 0x0000000b)

    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
    .short	5                               # DWARF version number
    .byte	5                               # DWARF Unit Type (DW_UT_split_compile)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	0
    .byte	1                               # Abbrev [1] 0x14:0x5 DW_TAG_compile_unit
    .long	0                               # DW_AT_macros
.Ldebug_info_dwo_end0:
    .section	.debug_macro.dwo,"e",@progbits
    .short	5                               # Macro information version
    .byte	0                               # Flags: 32 bit
    .byte	3                               # DW_MACRO_start_file
    .byte	0                               # Line Number
    .byte	0                               # File Number
    .byte	11                              # DW_MACRO_define_strx
    .byte	1                               # Line Number
    .byte	0                               # Macro String
    .byte	4                               # DW_MACRO_end_file
    .byte	0                               # End Of Macro List Mark
    .section	.debug_abbrev.dwo,"e",@progbits
    .byte	1                               # Abbreviation Code
    .byte	17                              # DW_TAG_compile_unit
    .byte	0                               # DW_CHILDREN_no
    .byte	121                             # DW_AT_macros
    .byte	23                              # DW_FORM_sec_offset
    .byte	0                               # EOM(1)
    .byte	0                               # EOM(2)
    .byte	0                               # EOM(3)
    .section	.debug_str.dwo,"eMS",@progbits,1
    .asciz	"x 5"                           # string offset=0
    .section	.debug_str_offsets.dwo,"e",@progbits
    .long	8                            # Length of String Offsets Set
    .short	5
    .short	0
  .long	0
