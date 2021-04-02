# RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data
t1_value equ 1 or 2

t1 BYTE t1_VALUE DUP (0)
; CHECK-LABEL: t1:
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .byte 0
; CHECK-NOT: .byte 0

t2_value equ 4 or t1_value
t2 BYTE t2_VALUE
; CHECK-LABEL: t2:
; CHECK-NEXT: .byte 7

t3_value equ t1_VALUE or 8
t3 BYTE t3_VALUE
; CHECK-LABEL: t3:
; CHECK-NEXT: .byte 11

END
