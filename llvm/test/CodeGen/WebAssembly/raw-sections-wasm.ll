; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown %s -o %t.o
; RUN: llvm-readobj --sections %t.o | FileCheck %s

; CHECK:      Type: CUSTOM
; CHECK-NEXT: Size: 4
; CHECK-NEXT: Offset:
; CHECK-NEXT: Name: __clangast

!0 = !{!"__clangast", i32 8, i32 0, !"\de\ad\be\ef"}
!llvm.raw.sections = !{!0}
