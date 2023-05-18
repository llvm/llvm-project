; RUN: llvm-mc -triple arm64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

; This is a regression test making sure we don't crash.

; CHECK:      Relocations [
; CHECK-NEXT:   Section __text {
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Offset: 0x0
; CHECK-NEXT:       PCRel: 0
; CHECK-NEXT:       Length: 2
; CHECK-NEXT:       Type: ARM64_RELOC_PAGEOFF12 (4)
; CHECK-NEXT:       Symbol: ltmp1
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]


        ldr     x0, [x8, L_bar@PAGEOFF]

        .section        __foo,__bar,regular,no_dead_strip
L_bar:
        .quad   0
