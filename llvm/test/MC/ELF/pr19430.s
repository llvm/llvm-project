// RUN: llvm-mc -triple x86_64-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj -r - | FileCheck %s

// Test that we can use .cfi_startproc without a global symbol.

.text
.space 1000
.cfi_startproc
 .cfi_endproc

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.eh_frame {
// CHECK-NEXT:     0x20 R_X86_64_PC32 .text 0x3E8
// CHECK-NEXT:   }
// CHECK-NEXT: ]
