// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck %s

// Test that this produces a R_X86_64_PLT32.

	jmp	foo@PLT

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[^ ]+}}) {{[^ ]+}} {
// CHECK-NEXT:     0x{{[^ ]+}} R_X86_64_PLT32 {{[^ ]+}} 0x{{[^ ]+}}
// CHECK-NEXT:   }
// CHECK-NEXT: ]
