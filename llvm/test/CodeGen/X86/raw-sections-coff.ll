; RUN: llc -filetype=obj -mtriple=x86_64-windows-msvc %s -o %t.o
; RUN: llvm-readobj --sections --section-data %t.o | FileCheck %s
; RUN: llc -mtriple=x86_64-windows-msvc %s -o %t.s
; RUN: FileCheck %s --check-prefix=ASM < %t.s
;; The assembly must round-trip to the same object contents.
; RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %t.s -o %t.mc.o
; RUN: llvm-readobj --sections --section-data %t.mc.o | FileCheck %s

; ASM:      .section clangast,"yD"
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\336\255\276\357"
; ASM:      .section bigalign,"yD"
; ASM-NEXT: .p2align 6, 0x0
; ASM-NEXT: .ascii "\001\002\003"
; ASM:      .section twoentry,"yD"
; ASM-NEXT: .p2align 2, 0x0
; ASM-NEXT: .ascii "\252\273\314"
; ASM:      .section twoentry,"yD"
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\021\""

; CHECK:      Name: clangast
; CHECK:      RawDataSize:
; CHECK:      Characteristics [
; CHECK-NEXT:   IMAGE_SCN_ALIGN_8BYTES
; CHECK-NEXT:   IMAGE_SCN_MEM_DISCARDABLE
; CHECK-NEXT: ]
; CHECK-NEXT: SectionData (
; CHECK-NEXT:   0000: DEADBEEF |
; CHECK-NEXT: )

; CHECK:      Name: bigalign
; CHECK:      RawDataSize: 0x3
; CHECK:      Characteristics [
; CHECK-NEXT:   IMAGE_SCN_ALIGN_64BYTES
; CHECK-NEXT:   IMAGE_SCN_MEM_DISCARDABLE
; CHECK-NEXT: ]
; CHECK-NEXT: SectionData (
; CHECK-NEXT:   0000: 010203 |
; CHECK-NEXT: )

; CHECK:      Name: twoentry
; CHECK:      RawDataSize: 0xA
; CHECK:      Characteristics [
; CHECK-NEXT:   IMAGE_SCN_ALIGN_8BYTES
; CHECK-NEXT:   IMAGE_SCN_MEM_DISCARDABLE
; CHECK-NEXT: ]
; CHECK-NEXT: SectionData (
; CHECK-NEXT:   0000: AABBCC00 00000000 1122 |
; CHECK-NEXT: )

!0 = !{!"clangast", i32 8, i32 0, !"\de\ad\be\ef"}
; Alignment larger than the default section alignment.
!1 = !{!"bigalign", i32 64, i32 0, !"\01\02\03"}
; Two entries in the same section. The second entry is aligned to 8, so 5 bytes
; of zero padding follow the 3 bytes of the first entry, and the section takes
; the larger of the two alignments.
!2 = !{!"twoentry", i32 4, i32 0, !"\aa\bb\cc"}
!3 = !{!"twoentry", i32 8, i32 0, !"\11\22"}
!llvm.raw.sections = !{!0, !1, !2, !3}
