; RUN: llc -filetype=obj -mtriple=x86_64-apple-darwin %s -o %t.o
; RUN: llvm-readobj --sections --section-data %t.o | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t.s
; RUN: FileCheck %s --check-prefix=ASM < %t.s
;; The assembly must round-trip to the same object contents.
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t.s -o %t.mc.o
; RUN: llvm-readobj --sections --section-data %t.mc.o | FileCheck %s

; ASM:      .section __CLANG,__clangast
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\336\255\276\357"
; ASM:      .section __CLANG,__bigalign
; ASM-NEXT: .p2align 6, 0x0
; ASM-NEXT: .ascii "\001\002\003"
; ASM:      .section __CLANG,__twoentries
; ASM-NEXT: .p2align 2, 0x0
; ASM-NEXT: .ascii "\252\273\314"
; ASM:      .section __CLANG,__twoentries
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\021\""

; CHECK:      Name: __clangast
; CHECK-NEXT: Segment: __CLANG
; CHECK:      Size:
; CHECK:      SectionData (
; CHECK-NEXT:   0000: DEADBEEF |
; CHECK-NEXT: )

; CHECK:      Name: __bigalign
; CHECK-NEXT: Segment: __CLANG
; CHECK:      Size: 0x3
; CHECK:      Alignment: 6
; CHECK:      SectionData (
; CHECK-NEXT:   0000: 010203 |
; CHECK-NEXT: )

; CHECK:      Name: __twoentries
; CHECK-NEXT: Segment: __CLANG
; CHECK:      Size: 0xA
; CHECK:      Alignment: 3
; CHECK:      SectionData (
; CHECK-NEXT:   0000: AABBCC00 00000000 1122 |
; CHECK-NEXT: )

!0 = !{!"__CLANG,__clangast", i32 8, i32 0, !"\de\ad\be\ef"}
; Alignment larger than the default section alignment.
!1 = !{!"__CLANG,__bigalign", i32 64, i32 0, !"\01\02\03"}
; Two entries in the same section. The second entry is aligned to 8, so 5 bytes
; of zero padding follow the 3 bytes of the first entry, and the section takes
; the larger of the two alignments.
!2 = !{!"__CLANG,__twoentries", i32 4, i32 0, !"\aa\bb\cc"}
!3 = !{!"__CLANG,__twoentries", i32 8, i32 0, !"\11\22"}
!llvm.raw.sections = !{!0, !1, !2, !3}
