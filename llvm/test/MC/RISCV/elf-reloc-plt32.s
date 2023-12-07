# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s

.globl func
func:

.data
.word extern_func@PLT - . + 4
.word func@PLT - . + 8

# CHECK:      Section ({{.*}}) .rela.data {
# CHECK-NEXT:   0x0 R_RISCV_PLT32 extern_func 0x4
# CHECK-NEXT:   0x4 R_RISCV_PLT32 func 0x8
# CHECK-NEXT: }
