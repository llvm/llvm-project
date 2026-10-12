; RUN: llc -filetype=obj -mtriple=x86_64-windows-msvc %s -o %t.o
; RUN: llvm-readobj --sections %t.o | FileCheck %s

; CHECK:      Name: clangast
; CHECK:      RawDataSize:
; CHECK:      Characteristics [
; CHECK-NEXT:   IMAGE_SCN_ALIGN_8BYTES
; CHECK-NEXT:   IMAGE_SCN_MEM_DISCARDABLE
; CHECK-NEXT: ]

!0 = !{!"clangast", i32 8, i32 0, !"\de\ad\be\ef"}
!llvm.raw.sections = !{!0}
