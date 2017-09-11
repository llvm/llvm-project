# RUN: llvm-dwarfdump %S/../Inputs/dwarfdump-ranges-baseaddr-exe.elf-x86-64 \
# RUN:  | FileCheck %s

## Executable binary for test produced from object built in 
## dwarfdump-ranges-baseaddr.s testcase.

# CHECK: .debug_info contents:
# CHECK: 0x0000000b: DW_TAG_compile_unit [1]
# CHECK:             DW_AT_low_pc [DW_FORM_addr]       (0x0000000000400078)
# CHECK-NEXT:        DW_AT_ranges [DW_FORM_sec_offset] (0x00000000
# CHECK-NEXT:    [0x0000000000400078 - 0x0000000000400079)
# CHECK-NEXT:    [0x000000000040007b - 0x000000000040007e)
# CHECK-NEXT:    [0x000000000040007f - 0x0000000000400080))
