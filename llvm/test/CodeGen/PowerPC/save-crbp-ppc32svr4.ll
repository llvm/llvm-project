; RUN: llc -relocation-model pic < %s | FileCheck %s

; CHECK-LABEL: fred
; CHECK: stwux 1, 1, 0
; Save R31..R29 via R0:
; CHECK: addic 0, 0, -4
; CHECK: stwx 31, 0, 0
; CHECK: addic 0, 0, -4
; CHECK: stwx 30, 0, 0
; CHECK: addic 0, 0, -4
; CHECK: stwx 29, 0, 0
; Set R29 back to the value of R0 from before the updates:
; CHECK: addic 29, 0, 12
; Save CR through R12 using R29 as the stack pointer (aligned base pointer).
; CHECK: mfcr 12
; CHECK: stw 28, -16(29)
; CHECK: stw 12, -20(29)

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-freebsd"

; Function Attrs: norecurse readnone sspstrong
define i64 @fred(double %a0) local_unnamed_addr #0 {
b1:
  %v2 = alloca i64, align 128
  store i64 0, i64* %v2
  %v3 = fcmp olt double %a0, 0x43E0000000000000
  br i1 %v3, label %b4, label %b8

b4:                                               ; preds = %b1
  %v5 = fcmp olt double %a0, 0xC3E0000000000000
  %v6 = fptosi double %a0 to i64
  store i64 %v6, i64* %v2
  %v7 = select i1 %v5, i64 -9223372036854775808, i64 %v6
  br label %b15

b8:                                               ; preds = %b1
  %v9 = fcmp olt double %a0, 0x43F0000000000000
  br i1 %v9, label %b10, label %b12

b10:                                              ; preds = %b8
  %v11 = fptoui double %a0 to i64
  br label %b15

b12:                                              ; preds = %b8
  %v13 = fcmp ogt double %a0, 0.000000e+00
  %v14 = sext i1 %v13 to i64
  br label %b15

b15:                                              ; preds = %b12, %b10, %b4
  %v16 = phi i64 [ %v7, %b4 ], [ %v11, %b10 ], [ %v14, %b12 ]
  %v17 = load i64, i64* %v2
  %v18 = add i64 %v17, %v16
  ret i64 %v18
}

attributes #0 = { norecurse readnone sspstrong "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "stack-protector-buffer-size"="8" "target-cpu"="ppc" }
