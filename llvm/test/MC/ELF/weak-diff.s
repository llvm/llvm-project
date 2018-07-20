// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu < %s | llvm-readobj -r | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.text {
// CHECK-NEXT:     0x1D R_X86_64_PLT32 f2 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.weak f
.weak g
f:
    nop
g:
    nop

.quad g - f


.weak f2
f2:
    nop
g2:
    nop
.quad g2 - f2
.quad f2 - g2
call f2
