; RUN: llvm-mc -triple arm64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

; Test that we produce an external relocation with Lbar. We could also produce
; an internal relocation. We just have to be careful to not use another symbol.

// CHECK:      Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x0
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: Lbar
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

        .section        __DATA,__objc_classrefs,regular,no_dead_strip
Lbar:

        .section        __DATA,__data
        .quad   Lbar

