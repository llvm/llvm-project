; RUN: llc -filetype=obj -mtriple=x86_64-linux-gnu %s -o %t.o
; RUN: llvm-readelf --sections -x __clangast -x __bigalign -x __twoentries %t.o | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu %s -o %t.s
; RUN: FileCheck %s --check-prefix=ASM < %t.s
;; The assembly must round-trip to the same object contents.
; RUN: llvm-mc -filetype=obj -triple=x86_64-linux-gnu %t.s -o %t.mc.o
; RUN: llvm-readelf --sections -x __clangast -x __bigalign -x __twoentries %t.mc.o | FileCheck %s

; ASM:      .section __clangast,"",@progbits
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\336\255\276\357"
; ASM:      .section __bigalign,"",@progbits
; ASM-NEXT: .p2align 6, 0x0
; ASM-NEXT: .ascii "\001\002\003"
; ASM:      .section __twoentries,"",@progbits
; ASM-NEXT: .p2align 2, 0x0
; ASM-NEXT: .ascii "\252\273\314"
; ASM:      .section __twoentries,"",@progbits
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .ascii "\021\""

; CHECK:      __clangast        PROGBITS  {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} 00      0   0  8
; CHECK-NEXT: __bigalign        PROGBITS  {{[0-9a-f]+}} {{[0-9a-f]+}} 000003 00      0   0 64
; CHECK-NEXT: __twoentries      PROGBITS  {{[0-9a-f]+}} {{[0-9a-f]+}} 00000a 00      0   0  8

; CHECK:      Hex dump of section '__clangast':
; CHECK-NEXT: 0x00000000 deadbeef ....
; CHECK:      Hex dump of section '__bigalign':
; CHECK-NEXT: 0x00000000 010203 ...
; CHECK:      Hex dump of section '__twoentries':
; CHECK-NEXT: 0x00000000 aabbcc00 00000000 1122 ........."

!0 = !{!"__clangast", i32 8, i32 0, !"\de\ad\be\ef"}
; Alignment larger than the default section alignment.
!1 = !{!"__bigalign", i32 64, i32 0, !"\01\02\03"}
; Two entries in the same section. The second entry is aligned to 8, so 5 bytes
; of zero padding follow the 3 bytes of the first entry, and the section takes
; the larger of the two alignments.
!2 = !{!"__twoentries", i32 4, i32 0, !"\aa\bb\cc"}
!3 = !{!"__twoentries", i32 8, i32 0, !"\11\22"}
!llvm.raw.sections = !{!0, !1, !2, !3}
