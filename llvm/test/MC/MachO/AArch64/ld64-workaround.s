; RUN: llvm-mc -triple arm64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

; Test that we produce an external relocation. This is a known and temporary bug
; in ld64, where it mishandles pointer-sized internal relocations. We should be
; able to remove this entirely soon.

// CHECK:      Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x18
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: Llit16
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x10
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: Llit8
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x8
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: Llit4
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x0
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: Lcfstring
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

        .section        __DATA,__cfstring
Lcfstring:

        .section        __DATA,__literal4,4byte_literals
Llit4:
        .word 42

        .section        __DATA,__literal8,8byte_literals
Llit8:
        .quad 42

        .section        __DATA,__literal16,16byte_literals
Llit16:
        .quad 42
        .quad 42

        .section        __DATA,__data
        .quad   Lcfstring
        .quad   Llit4
        .quad   Llit8
        .quad   Llit16
