# RUN: not llvm-mc -filetype=obj -triple=ve %s -o /dev/null 2>&1 | \
# RUN:     FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple=ve -position-independent %s \
# RUN:     -o /dev/null 2>&1 | FileCheck %s

.data
a:
## An undefined reference of _GLOBAL_OFFSET_TABLE_ causes .got[0] to be
## allocated to store _DYNAMIC.
.byte _GLOBAL_OFFSET_TABLE_
.byte _GLOBAL_OFFSET_TABLE_ - .
.2byte _GLOBAL_OFFSET_TABLE_
.2byte _GLOBAL_OFFSET_TABLE_ - .
.8byte _GLOBAL_OFFSET_TABLE_ - .

# CHECK:      data-reloc-error.s:10:7: error: 1-byte data relocation is not supported
# CHECK-NEXT: .byte _GLOBAL_OFFSET_TABLE_
# CHECK:      data-reloc-error.s:11:7: error: 1-byte pc-relative data relocation is not supported
# CHECK-NEXT: .byte _GLOBAL_OFFSET_TABLE_ - .
# CHECK:      data-reloc-error.s:12:8: error: 2-byte data relocation is not supported
# CHECK-NEXT: .2byte _GLOBAL_OFFSET_TABLE_
# CHECK:      data-reloc-error.s:13:8: error: 2-byte pc-relative data relocation is not supported
# CHECK-NEXT: .2byte _GLOBAL_OFFSET_TABLE_ - .
# CHECK:      data-reloc-error.s:14:8: error: 8-byte pc-relative data relocation is not supported
# CHECK-NEXT: .8byte _GLOBAL_OFFSET_TABLE_ - .

