// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o - | llvm-readobj -r - | FileCheck %s
// RUN: llvm-mc -filetype=obj -relax-relocations=false -triple x86_64-pc-linux %s -o - | llvm-readobj -r - | FileCheck --check-prefix=OLD %s

// these should produce R_X86_64_GOTPCRELX

        call *call@GOTPCREL(%rip)
        jmp *jmp@GOTPCREL(%rip)

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.text {
// CHECK-NEXT:     R_X86_64_GOTPCRELX call
// CHECK-NEXT:     R_X86_64_GOTPCRELX jmp
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// OLD:      Relocations [
// OLD-NEXT:   Section ({{.*}}) .rela.text {
// OLD-NEXT:     R_X86_64_GOTPCREL call
// OLD-NEXT:     R_X86_64_GOTPCREL jmp
// OLD-NEXT:   }
// OLD-NEXT: ]
