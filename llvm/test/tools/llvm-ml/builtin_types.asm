; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

t1_long BYTE 1
t1_short DB 1
t1_signed SBYTE -1

; CHECK-LABEL: t1_long:
; CHECK: .byte 1
; CHECK-LABEL: t1_short:
; CHECK: .byte 1
; CHECK-LABEL: t1_signed:
; CHECK: .byte -1

t2_long WORD 2
t2_short DW 2
t2_signed SWORD -2

; CHECK-LABEL: t2_long:
; CHECK: .short 2
; CHECK-LABEL: t2_short:
; CHECK: .short 2
; CHECK-LABEL: t2_signed:
; CHECK: .short -2

t3_long DWORD 3
t3_short DD 3
t3_signed SDWORD -3

; CHECK-LABEL: t3_long:
; CHECK: .long 3
; CHECK-LABEL: t3_short:
; CHECK: .long 3
; CHECK-LABEL: t3_signed:
; CHECK: .long -3

t4_long FWORD 4
t4_short DF 4
t4_long_large FWORD 4294967298
t4_short_large FWORD 4294967298

; CHECK-LABEL: t4_long:
; CHECK-NEXT: .long 4
; CHECK-NEXT: .short 0
; CHECK-LABEL: t4_short:
; CHECK-NEXT: .long 4
; CHECK-NEXT: .short 0
; CHECK-LABEL: t4_long_large:
; CHECK-NEXT: .long 2
; CHECK-NEXT: .short 1
; CHECK-LABEL: t4_short_large:
; CHECK-NEXT: .long 2
; CHECK-NEXT: .short 1

t5_long QWORD 4611686018427387904
t5_short DQ 4611686018427387904
t5_signed SQWORD -4611686018427387904

; CHECK-LABEL: t5_long:
; CHECK-NEXT: .quad 4611686018427387904
; CHECK-LABEL: t5_short:
; CHECK-NEXT: .quad 4611686018427387904
; CHECK-LABEL: t5_signed:
; CHECK-NEXT: .quad -4611686018427387904

t6_single REAL4 1.3
t6_single_hex REAL4 3fa66666r

; CHECK-LABEL: t6_single:
; CHECK-NEXT: .long 1067869798
; CHECK-LABEL: t6_single_hex:
; CHECK-NEXT: .long 1067869798

t7_double REAL8 1.3
t7_double_hex REAL8 3FF4CCCCCCCCCCCDR

; CHECK-LABEL: t7_double:
; CHECK-NEXT: .quad 4608533498688228557
; CHECK-LABEL: t7_double_hex:
; CHECK-NEXT: .quad 4608533498688228557

t8_extended REAL10 1.3
t8_extended_hex REAL10 3FFFA666666666666666r

; CHECK-LABEL: t8_extended:
; CHECK-NEXT: .ascii "fffffff\246\377?"
; CHECK-LABEL: t8_extended_hex:
; CHECK-NEXT: .ascii "fffffff\246\377?"

.code

END
