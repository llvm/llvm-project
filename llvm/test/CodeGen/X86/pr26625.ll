; RUN: llc < %s -mcpu=i686 2>&1 | FileCheck %s
; PR26625

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386"

define float @x0(float %f) #0 {
entry:
  %call = tail call float @sqrtf(float %f) #1
  ret float %call
; CHECK-LABEL: x0:
; CHECK: flds
; CHECK-NEXT: fsqrt
; CHECK-NOT: vsqrtss
}

declare float @sqrtf(float) #0

attributes #0 = { nounwind optsize readnone }
attributes #1 = { nounwind optsize readnone }
