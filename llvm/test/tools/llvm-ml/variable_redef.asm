; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

; <var> = <expression> can be redefined at any time.
assigned_number = 1
t1_original BYTE assigned_number
assigned_number = 1
t1_reset BYTE assigned_number
assigned_number = 2
t1_changed BYTE assigned_number

; CHECK-LABEL: t1_original:
; CHECK-NEXT: .byte 1

; CHECK-LABEL: t1_reset:
; CHECK-NEXT: .byte 1

; CHECK-LABEL: t1_changed:
; CHECK-NEXT: .byte 2

; <var> EQU <expression> can be redundantly set, but can't be changed.
equated_number equ 3
t2_original BYTE equated_number
equated_number equ 3
t2_reset BYTE equated_number

; CHECK-LABEL: t2_original:
; CHECK-NEXT: .byte 3

; CHECK-LABEL: t2_reset:
; CHECK-NEXT: .byte 3

; <var> EQU <text> can be redefined at any time.
equated_text equ <4, 5>
t3_original BYTE equated_text
equated_text equ <4, 5>
t3_reset BYTE equated_text
equated_text equ <5, 6>
t3_changed BYTE equated_text

; CHECK-LABEL: t3_original:
; CHECK-NEXT: .byte 4
; CHECK-NEXT: .byte 5

; CHECK-LABEL: t3_reset:
; CHECK-NEXT: .byte 4
; CHECK-NEXT: .byte 5

; CHECK-LABEL: t3_changed:
; CHECK-NEXT: .byte 5
; CHECK-NEXT: .byte 6

; <var> TEXTEQU <text> can be redefined at any time.
textequated_text textequ <7, 8>
t4_original BYTE textequated_text
textequated_text textequ <7, 8>
t4_reset BYTE textequated_text
textequated_text textequ <9, 10>
t4_changed BYTE textequated_text

; CHECK-LABEL: t4_original:
; CHECK-NEXT: .byte 7
; CHECK-NEXT: .byte 8

; CHECK-LABEL: t4_reset:
; CHECK-NEXT: .byte 7
; CHECK-NEXT: .byte 8

; CHECK-LABEL: t4_changed:
; CHECK-NEXT: .byte 9
; CHECK-NEXT: .byte 10

.code

end
