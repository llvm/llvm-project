; RUN: llc -disable-ppc-vsx-fma-mutation=false -mcpu=pwr10 -verify-machineinstrs \
; RUN:   -ppc-asm-full-reg-names -mtriple powerpc64-ibm-aix7.2.0.0 < %s | FileCheck %s 

target datalayout = "E-m:a-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"

define void @initial(<2 x double> %0){
entry:
  %1 = fmul <2 x double> %0, zeroinitializer
  br label %for.cond251.preheader.lr.ph

for.cond251.preheader.lr.ph:                      ; preds = %for.cond251.preheader.lr.ph, %entry
  %2 = phi double [ %3, %for.cond251.preheader.lr.ph ], [ 0.000000e+00, %entry ]
  %3 = phi double [ %7, %for.cond251.preheader.lr.ph ], [ 0.000000e+00, %entry ]
  %add737 = fadd double %3, %2
  %4 = insertelement <2 x double> zeroinitializer, double %add737, i64 0
  %5 = fmul contract <2 x double> %4, zeroinitializer
  %6 = fadd contract <2 x double> %1, %5
  %7 = extractelement <2 x double> %6, i64 0
  br label %for.cond251.preheader.lr.ph
}

; CHECK:        xsadddp f4, f3, f4
; CHECK-NEXT:   xxmrghd vs5, vs4, vs2
; CHECK-NEXT:   fmr f4, f3
; CHECK-NEXT:   xvmaddmdp vs5, vs0, vs1
; CHECK-NEXT:   fmr f3, f5
