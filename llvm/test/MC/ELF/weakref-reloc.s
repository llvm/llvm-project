// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r -t | FileCheck %s

// Test that the relocations point to the correct symbols.

	.weakref bar,foo
        call    zed@PLT
     call	bar

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) {{[^ ]+}} {
// CHECK-NEXT:     0x1 R_X86_64_PLT32 zed 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x6 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:   }
// CHECK-NEXT: ]
