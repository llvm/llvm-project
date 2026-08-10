; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown %s -o %t.o
; RUN: llvm-readobj --sections %t.o | FileCheck %s

; CHECK: Name: __clangast

!0 = !{!"__clangast", i32 8, i32 1, !"\de\ad\be\ef"}
!llvm.raw.sections = !{!0}
