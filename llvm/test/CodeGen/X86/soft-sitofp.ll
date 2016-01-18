; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-pc-linux"

; Function Attrs: nounwind
; CHECK-LABEL: ll_to_d:
; CHECK: calll __floatdidf
define double @ll_to_d(i64 %n) #0 {
entry:
  %conv = sitofp i64 %n to double
  ret double %conv
}

attributes #0 = { nounwind "use-soft-float"="true" }
