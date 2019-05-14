// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r --symbols | FileCheck %s

// Test that this produces the correct relocations R_X86_64_GOT32 and that we,
// unlike gas, don't create a _GLOBAL_OFFSET_TABLE_ symbol as a side effect.

        movl	foo@GOT, %eax
        movl	foo@GOTPCREL(%rip), %eax

// CHECK:      Relocations [
// CHECK:        Section ({{[^ ]+}}) .rela.text {
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_GOT32 foo 0x{{[^ ]+}}
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_GOTPCREL foo 0x{{[^ ]+}}
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:        Symbols [
// CHECK-NOT:          _GLOBAL_OFFSET_TABLE_
