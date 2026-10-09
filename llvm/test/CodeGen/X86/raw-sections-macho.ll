; RUN: llc -filetype=obj -mtriple=x86_64-apple-darwin %s -o %t.o
; RUN: llvm-readobj --sections %t.o | FileCheck %s

; CHECK:      Name: __clangast
; CHECK-NEXT: Segment: __CLANG
; CHECK:      Size:

!0 = !{!"__CLANG,__clangast", i32 8, i32 1, !"\de\ad\be\ef"}
!llvm.raw.sections = !{!0}
