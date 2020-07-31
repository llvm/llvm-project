// COFF section-relative relocations

// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj -r - | FileCheck %s

.data
values:
    .long 1
    .long 0

.text
    movq    values@SECREL32(%rax), %rcx
    movq    values@SECREL32+8(%rax), %rax

// CHECK:      Relocations [
// CHECK-NEXT:   Section (1) .text {
// CHECK-NEXT:     0x3 IMAGE_REL_AMD64_SECREL values
// CHECK-NEXT:     0xA IMAGE_REL_AMD64_SECREL values
// CHECK-NEXT:   }
// CHECK-NEXT: ]
