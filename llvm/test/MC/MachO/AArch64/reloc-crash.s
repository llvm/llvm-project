; RUN: llvm-mc -triple arm64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

; Test tha we produce an external relocation. There is no apparent need for it, but
; ld64 (241.9) crashes if we don't.

; CHECK:      Relocations [
; CHECK-NEXT:   Section __bar {
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x0
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 3
; CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
; CHECK-NEXT:       Symbol: Lbar
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]

        .section        __TEXT,__cstring
Lfoo:
        .asciz  "Hello World!"
Lbar:
        .asciz  "cString"

        .section        __foo,__bar,literal_pointers
        .quad   Lbar
