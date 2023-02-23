# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s

.data
.word extern_func@PLT - . + 4

# CHECK:      Section ({{.*}}) .rela.data {
# CHECK-NEXT:   0x0 R_RISCV_PLT32 extern_func 0x4
# CHECK-NEXT: }
