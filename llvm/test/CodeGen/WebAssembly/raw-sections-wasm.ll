; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown %s -o %t.o
; RUN: llvm-readobj --sections --section-data %t.o | FileCheck %s
; RUN: llc -mtriple=wasm32-unknown-unknown %s -o %t.s
; RUN: FileCheck %s --check-prefix=ASM < %t.s
;; The assembly must round-trip to the same object contents.
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t.s -o %t.mc.o
; RUN: llvm-readobj --sections --section-data %t.mc.o | FileCheck %s

;; In assembly, custom sections carry a .custom_section. prefix so that they
;; are not mistaken for data sections.
; ASM:      .section .custom_section.__clangast,"",@
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\336\255\276\357"
; ASM:      .section .custom_section.__bigalign,"",@
; ASM-NEXT: .p2align 6, 0x0
; ASM-NEXT: .ascii "\001\002\003"
; ASM:      .section .custom_section.__twoentries,"",@
; ASM-NEXT: .p2align 2, 0x0
; ASM-NEXT: .ascii "\252\273\314"
; ASM:      .section .custom_section.__twoentries,"",@
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\021\""

; CHECK:      Type: CUSTOM
; CHECK-NEXT: Size: 4
; CHECK-NEXT: Offset:
; CHECK-NEXT: Name: __clangast
; CHECK-NEXT: SectionData (
; CHECK-NEXT:   0000: DEADBEEF |
; CHECK-NEXT: )

; Wasm custom sections have no alignment, so only the data is checked.
; CHECK:      Type: CUSTOM
; CHECK-NEXT: Size: 3
; CHECK-NEXT: Offset:
; CHECK-NEXT: Name: __bigalign
; CHECK-NEXT: SectionData (
; CHECK-NEXT:   0000: 010203 |
; CHECK-NEXT: )

; The padding between entries is relative to the start of the section payload.
; CHECK:      Type: CUSTOM
; CHECK-NEXT: Size: 10
; CHECK-NEXT: Offset:
; CHECK-NEXT: Name: __twoentries
; CHECK-NEXT: SectionData (
; CHECK-NEXT:   0000: AABBCC00 00000000 1122 |
; CHECK-NEXT: )

!0 = !{!"__clangast", i32 8, i32 0, !"\de\ad\be\ef"}
; Alignment larger than the default section alignment.
!1 = !{!"__bigalign", i32 64, i32 0, !"\01\02\03"}
; Two entries in the same section. The second entry is aligned to 8, so 5 bytes
; of zero padding follow the 3 bytes of the first entry.
!2 = !{!"__twoentries", i32 4, i32 0, !"\aa\bb\cc"}
!3 = !{!"__twoentries", i32 8, i32 0, !"\11\22"}
!llvm.raw.sections = !{!0, !1, !2, !3}
