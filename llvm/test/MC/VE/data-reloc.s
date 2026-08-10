# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | \
# RUN:     llvm-objdump -r - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj -position-independent %s -o - | \
# RUN:     llvm-objdump -r - | FileCheck %s

.data
a:
## An undefined reference of _GLOBAL_OFFSET_TABLE_ causes .got[0] to be
## allocated to store _DYNAMIC.
.4byte _GLOBAL_OFFSET_TABLE_ - 8
.4byte _GLOBAL_OFFSET_TABLE_ - . - 8
.8byte _GLOBAL_OFFSET_TABLE_ - 8

# CHECK: 0000000000000000 R_VE_REFLONG _GLOBAL_OFFSET_TABLE_-0x8
# CHECK: 0000000000000004 R_VE_SREL32 _GLOBAL_OFFSET_TABLE_-0x8
# CHECK: 0000000000000008 R_VE_REFQUAD _GLOBAL_OFFSET_TABLE_-0x8
