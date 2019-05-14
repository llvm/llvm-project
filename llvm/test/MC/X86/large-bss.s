// RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o - | llvm-readobj -S | FileCheck %s

.bss
.zero 0x10000000000000

// CHECK:      Name: .bss
// CHECK-NEXT: Type: SHT_NOBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0
// CHECK-NEXT: Offset: 0x40
// CHECK-NEXT: Size: 4503599627370496
