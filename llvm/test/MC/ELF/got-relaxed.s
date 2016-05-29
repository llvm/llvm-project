// RUN: llvm-mc -filetype=obj -relax-relocations -triple x86_64-pc-linux %s -o - | llvm-readobj -r | FileCheck %s

// these should produce R_X86_64_GOTPCRELX

        call *call@GOTPCREL(%rip)
        jmp *jmp@GOTPCREL(%rip)

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.text {
// CHECK-NEXT:     R_X86_64_GOTPCRELX call
// CHECK-NEXT:     R_X86_64_GOTPCRELX jmp
// CHECK-NEXT:   }
// CHECK-NEXT: ]
