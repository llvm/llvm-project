; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

; TEXTEQU with text macro argument should expand correctly.
part1 TEXTEQU <1>
part2 TEXTEQU <0>
joined TEXTEQU part1, part2
t1 BYTE joined

; CHECK-LABEL: t1:
; CHECK-NEXT: .byte 10

; TEXTEQU text-list with 3 elements.
a2 TEXTEQU <1>
b2 TEXTEQU <2>
c2 TEXTEQU <3>
triple TEXTEQU a2, b2, c2
t2 BYTE triple

; CHECK-LABEL: t2:
; CHECK-NEXT: .byte 123

; TEXTEQU where a later element is a text macro.
; The text macro should be resolved by parseTextItem, not the lexer.
inner TEXTEQU <5>
outer TEXTEQU <2>, inner
t3 BYTE outer

; CHECK-LABEL: t3:
; CHECK-NEXT: .byte 25

; TEXTEQU with trailing ;;
ts4 TEXTEQU <42>;;
t4 BYTE ts4

; CHECK-LABEL: t4:
; CHECK-NEXT: .byte 42

; TEXTEQU multi-element with trailing ;;
a5 TEXTEQU <1>
b5 TEXTEQU <0>
ts5 TEXTEQU a5, b5;;
t5 BYTE ts5

; CHECK-LABEL: t5:
; CHECK-NEXT: .byte 10

; TEXTEQU text-list spanning multiple lines.
a6 TEXTEQU <1>
b6 TEXTEQU <0>
ts6 TEXTEQU a6,
  b6
t6 BYTE ts6

; CHECK-LABEL: t6:
; CHECK-NEXT: .byte 10

; TEXTEQU text-list with ;; between items.
a7 TEXTEQU <1>
b7 TEXTEQU <0>
ts7 TEXTEQU a7,;;
  b7
t7 BYTE ts7

; CHECK-LABEL: t7:
; CHECK-NEXT: .byte 10

; EQU with text literal creates a text macro.
eq8 EQU <42>
t8 BYTE eq8

; CHECK-LABEL: t8:
; CHECK-NEXT: .byte 42

; EQU with numeric expression creates a numeric constant.
eq9 EQU 7
t9 DWORD eq9

; CHECK-LABEL: t9:
; CHECK-NEXT: .long 7

; EQU with text macro identifier: the text macro is expanded by the lexer,
; then evaluated as a numeric expression (not stored as a text macro).
txt10 TEXTEQU <99>
eq10 EQU txt10
t10 DWORD eq10

; CHECK-LABEL: t10:
; CHECK-NEXT: .long 99

end
