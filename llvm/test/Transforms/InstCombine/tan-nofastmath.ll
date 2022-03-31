; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define float @mytan(float %x) {
entry:
  %call = call float @atanf(float %x)
  %call1 = call float @tanf(float %call) 
  ret float %call1
}

; CHECK-LABEL: define float @mytan(
; CHECK:   %call = call float @atanf(float %x)
; CHECK-NEXT:   %call1 = call float @tanf(float %call)
; CHECK-NEXT:   ret float %call1
; CHECK-NEXT: }

declare float @tanf(float)
declare float @atanf(float)
