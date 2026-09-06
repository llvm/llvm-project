; RUN: llc -filetype=obj -mtriple=x86_64-linux-gnu %s -o %t.o
; RUN: llvm-readelf --sections %t.o | FileCheck %s

; CHECK: __clangast        PROGBITS  {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} 00   A  0   0  8

!0 = !{!"__clangast", i32 8, i32 1, !"\de\ad\be\ef"}
!llvm.raw.sections = !{!0}
