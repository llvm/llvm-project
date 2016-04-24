// RUN: llvm-mc -filetype=obj -relax-relocations -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r -t | FileCheck %s

// Test that this produces the correct relaxed relocations.

        movl	foo@GOT, %eax
        movl	foo@GOTPCREL(%rip), %eax
        movq  foo@GOTPCREL(%rip), %rax
        .long zed@GOTPCREL

// CHECK:      Relocations [
// CHECK:        Section ({{[^ ]+}}) .rela.text {
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_GOT32 foo 0x{{[^ ]+}}
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_GOTPCRELX foo 0x{{[^ ]+}}
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_REX_GOTPCRELX foo 0x{{[^ ]+}}
// CHECK-NEXT:       0x{{[^ ]+}} R_X86_64_GOTPCREL zed 0x{{[^ ]+}}
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:        Symbols [
// CHECK-NOT:          _GLOBAL_OFFSET_TABLE_
