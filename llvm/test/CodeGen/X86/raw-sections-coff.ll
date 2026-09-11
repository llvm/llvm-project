; RUN: llc -filetype=obj -mtriple=x86_64-windows-msvc %s -o %t.o
; RUN: llvm-readobj --sections %t.o | FileCheck %s

; CHECK:      Name: clangast
; CHECK:      RawDataSize:
; CHECK:      Characteristics [
; CHECK-DAG:    IMAGE_SCN_CNT_INITIALIZED_DATA
; CHECK-DAG:    IMAGE_SCN_MEM_READ
; CHECK:      ]

!0 = !{!"clangast", i32 8, i32 1, !"\de\ad\be\ef"}
!llvm.raw.sections = !{!0}
