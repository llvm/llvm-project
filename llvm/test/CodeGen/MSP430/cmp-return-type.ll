; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430"

define i16 @f(float %a, float %b) optsize {
  %cmp = fcmp une float %a, %b
  %conv = zext i1 %cmp to i16
; CHECK-LABEL: call #__mspabi_cmpf
; CHECK-NOT:   r13
; CHECK-LABEL: mov r2

; This is quite fragile attempt to detect the return type:
; Correct:
;        call    #__mspabi_cmpf
;        tst     r12
;        mov     r2, r13
; Incorrect:
;        call    #__mspabi_cmpf
;        bis     r12, r13        <-- checking (R12:R13)
;        tst     r13
;        mov     r2, r13

  ret i16 %conv
}
