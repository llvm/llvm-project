// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj  --symbols -r --expand-relocs | FileCheck %s

// Test that we can forward reference a section.

mov .rodata, %rsi
.section .rodata

// CHECK:Relocations [
// CHECK:  Section {{.*}} .rela.text {
// CHECK:    Relocation {
// CHECK:      Offset: 0x4
// CHECK:      Type: R_X86_64_32S (11)
// CHECK:      Symbol: .rodata
// CHECK:      Addend: 0x0
// CHECK:    }
// CHECK:  }
// CHECK:]

// There is only one .rodata symbol

// CHECK:Symbols [
// CHECK:   Type: Section (0x3)
// CHECK:   Section: .rodata
// CHECK-NOT:   Section: .rodata
