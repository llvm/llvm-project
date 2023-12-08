// RUN: not llvm-mc -triple x86_64-pc-linux %s -o %t.o -filetype=obj 2>&1 | FileCheck %s

        .quad foo@gotpcrel
// CHECK:      32 bit reloc applied to a field with a different size
// CHECK-NEXT: .quad foo@gotpcrel

        .quad foo@plt
// CHECK:      32 bit reloc applied to a field with a different size
// CHECK-NEXT: .quad foo@plt

        .quad foo@tlsld
// CHECK:      32 bit reloc applied to a field with a different size
// CHECK-NEXT: .quad foo@tlsld

        .quad foo@gottpoff
// CHECK:      32 bit reloc applied to a field with a different size
// CHECK-NEXT: .quad foo@gottpoff

        .quad foo@tlsgd
// CHECK:      32 bit reloc applied to a field with a different size
// CHECK-NEXT: .quad foo@tlsgd
