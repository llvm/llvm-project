; RUN: llc -relocation-model pic < %s | FileCheck %s
;
; Make sure that the CR register is saved correctly on PPC32/SVR4.

; CHECK-LABEL: fred:
; CHECK: stwu 1, -48(1)
; CHECK: stw 31, 36(1)
; CHECK: mr 31, 1
; CHECK-DAG: stw 30, 32(1)
; CHECK-DAG: mfcr [[CR:[0-9]+]]
; CHECK: stw [[CR]], 28(31)

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-freebsd"

; Function Attrs: norecurse nounwind readnone sspstrong
define i64 @fred(double %a0) local_unnamed_addr #0 {
b1:
  %a1 = tail call double asm "fadd $0, $1, $2", "=f,f,f,~{cr2}"(double %a0, double %a0)
  %v2 = fcmp olt double %a1, 0x43E0000000000000
  br i1 %v2, label %b3, label %b7

b3:                                               ; preds = %b1
  %v4 = fcmp olt double %a0, 0xC3E0000000000000
  %v5 = fptosi double %a0 to i64
  %v6 = select i1 %v4, i64 -9223372036854775808, i64 %v5
  br label %b14

b7:                                               ; preds = %b1
  %v8 = fcmp olt double %a0, 0x43F0000000000000
  br i1 %v8, label %b9, label %b11

b9:                                               ; preds = %b7
  %v10 = fptoui double %a0 to i64
  br label %b14

b11:                                              ; preds = %b7
  %v12 = fcmp ogt double %a0, 0.000000e+00
  %v13 = sext i1 %v12 to i64
  br label %b14

b14:                                              ; preds = %b11, %b9, %b3
  %v15 = phi i64 [ %v6, %b3 ], [ %v10, %b9 ], [ %v13, %b11 ]
  ret i64 %v15
}

attributes #0 = { norecurse nounwind readnone sspstrong "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "stack-protector-buffer-size"="8" "target-cpu"="ppc" }
