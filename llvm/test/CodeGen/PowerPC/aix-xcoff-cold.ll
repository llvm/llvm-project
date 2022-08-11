; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-objdump --syms %t.o | FileCheck %s

; CHECK:      SYMBOL TABLE:
; CHECK-NEXT: 0000000000000000      df *DEBUG* 0000000000000000 <stdin>
; CHECK-NEXT: 0000000000000000 l       .text   000000000000001e .text
; CHECK-NEXT: 0000000000000000 g     F .text (csect: .text)  0000000000000000 .cold_fun
; CHECK-NEXT: 0000000000000020 g     O .data   0000000000000018 cold_fun

define dso_local void @cold_fun() #1 {
entry:
  ret void
}

attributes #1 = { cold }
