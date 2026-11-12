# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.dwp
# RUN: not llvm-dwp %t.dwp -o /dev/null 2>&1 | FileCheck %s

## Note: To reach the test point, we need to use comdat groups, to have multiple
## .debug_info sections. One comdat group needs to have one complete unit header,
## the second one may be completely empty. 
## Furthermore, the .debug_cu_index also does not need to be complete.

# CHECK: error: expected exactly one occurrence of a debug info section in a .dwp file
    .section	.debug_info.dwo,"G",@progbits,0xFDFDFDFD,comdat
    .long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
    .short	5                               # DWARF version number
    .byte	5                               # DWARF Unit Type (DW_UT_split_compile)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	-1506010254921578184
    .byte	1                               # Abbrev [1] DW_TAG_compile_unit
.Ldebug_info_dwo_end1:
.section	.debug_info.dwo,"G",@progbits,0xDFDFDFDF,comdat
    .long	0                               # Length of Unit
  .section .debug_cu_index, "", @progbits
## Incomplete Header:
    .long 2                         # Version
