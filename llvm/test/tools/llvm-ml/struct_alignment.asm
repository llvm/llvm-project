; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

FOO STRUCT 8
  f FWORD -1
FOO ENDS

t1 FOO <>
; CHECK-LABEL: t1:
; CHECK-NEXT: .long 4294967295
; CHECK-NEXT: .short 65535
; CHECK-NOT: .zero

BAZ STRUCT
  b BYTE 3 DUP (-1)
  f FWORD -1
BAZ ENDS

FOOBAR STRUCT 8
  f1 BAZ <>
  f2 BAZ <>
  h BYTE -1
FOOBAR ENDS

t2 FOOBAR <>
; CHECK-LABEL: t2:
; CHECK-NEXT: .byte -1
; CHECK-NEXT: .byte -1
; CHECK-NEXT: .byte -1
; CHECK-NEXT: .long 4294967295
; CHECK-NEXT: .short 65535
; CHECK-NEXT: .zero 3
; CHECK-NEXT: .byte -1
; CHECK-NEXT: .byte -1
; CHECK-NEXT: .byte -1
; CHECK-NEXT: .long 4294967295
; CHECK-NEXT: .short 65535
; CHECK-NEXT: .byte -1
; CHECK-NEXT: .zero 2

.code

END
