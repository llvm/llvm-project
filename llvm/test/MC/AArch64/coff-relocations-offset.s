// RUN: llvm-mc -triple aarch64-windows -filetype obj -o %t.obj %s
// RUN: llvm-objdump -d -r %t.obj | FileCheck %s
// RUN: llvm-readobj --syms %t.obj | FileCheck %s --check-prefix=SYMBOLS

    .text
main:
    adrp x0, .Ltmp0
    adrp x0, .Ltmp1
    adrp x0, .Ltmp2+8

    .section .rdata
    .word 1
.Ltmp0:
    .word 2
    .fill 1048576
.Ltmp1: // 1 MB + 8 bytes
    .fill (1048576-8-4)
.Ltmp2: // 2 MB - 4 bytes
    .word 3
    // 2 MB here
    .word 4
    // .Ltmp2+8 points here
    .word 5

// CHECK:      0: 20 00 00 90   adrp    x0, 0x4000
// CHECK-NEXT:          0000000000000000:  IMAGE_REL_ARM64_PAGEBASE_REL21       .rdata
// CHECK-NEXT: 4: 40 00 00 90   adrp    x0, 0x8000
// CHECK-NEXT:          0000000000000004:  IMAGE_REL_ARM64_PAGEBASE_REL21       $L.rdata_1
// CHECK-NEXT: 8: 20 00 00 90   adrp    x0, 0x4000
// CHECK-NEXT:          0000000000000008:  IMAGE_REL_ARM64_PAGEBASE_REL21       $L.rdata_2

// SYMBOLS:      Symbol {
// SYMBOLS:        Name: $L.rdata_1
// SYMBOLS-NEXT:   Value: 1048576
// SYMBOLS-NEXT:   Section: .rdata (4)
// SYMBOLS-NEXT:   BaseType: Null (0x0)
// SYMBOLS-NEXT:   ComplexType: Null (0x0)
// SYMBOLS-NEXT:   StorageClass: Label (0x6)
// SYMBOLS-NEXT:   AuxSymbolCount: 0
// SYMBOLS-NEXT: }
// SYMBOLS-NEXT: Symbol {
// SYMBOLS-NEXT:   Name: $L.rdata_2
// SYMBOLS-NEXT:   Value: 2097152
// SYMBOLS-NEXT:   Section: .rdata (4)
// SYMBOLS-NEXT:   BaseType: Null (0x0)
// SYMBOLS-NEXT:   ComplexType: Null (0x0)
// SYMBOLS-NEXT:   StorageClass: Label (0x6)
// SYMBOLS-NEXT:   AuxSymbolCount: 0
// SYMBOLS-NEXT: }
