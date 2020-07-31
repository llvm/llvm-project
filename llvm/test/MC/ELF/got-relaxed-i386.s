// RUN: llvm-mc -filetype=obj -triple i386-pc-linux %s -o - | llvm-readobj -r - | FileCheck %s
// RUN: llvm-mc -filetype=obj -relax-relocations=false -triple i386-pc-linux %s -o - | llvm-readobj -r - | FileCheck --check-prefix=OLD %s

        movl mov@GOT(%ebx), %eax
        mull mul@GOT(%ebx)
        .long long@GOT

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.text {
// CHECK-NEXT:     R_386_GOT32X mov
// CHECK-NEXT:     R_386_GOT32 mul
// CHECK-NEXT:     R_386_GOT32 long
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// OLD:      Relocations [
// OLD-NEXT:   Section ({{.*}}) .rel.text {
// OLD-NEXT:     R_386_GOT32 mov
// OLD-NEXT:     R_386_GOT32 mul
// OLD-NEXT:     R_386_GOT32 long
// OLD-NEXT:   }
// OLD-NEXT: ]
