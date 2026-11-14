; RUN: llc < %s -O1 -mtriple=sparc -mcpu=leon3 | FileCheck %s

;; The scheduling model comes from the tune CPU. All functions have the same
;; CPU and features, so each must get a subtarget for its own tune CPU and not
;; reuse the one created for the function before it.

; CHECK-LABEL: no_tune_before:
; CHECK:       ld
; CHECK-NEXT:  ld
; CHECK-NEXT:  ld
; CHECK-NEXT:  fadds
; CHECK-NEXT:  ld
; CHECK-NEXT:  fsubs
; CHECK-NEXT:  fmuls
; CHECK-NEXT:  retl
; CHECK-NEXT:  fdivs

define float @no_tune_before(ptr byval(float) %a, ptr byval(float) %b, ptr byval(float) %c, ptr byval(float) %d) {
entry:
  %0 = load float, ptr %a, align 8
  %1 = load float, ptr %b, align 8
  %2 = load float, ptr %c, align 8
  %3 = load float, ptr %d, align 8
  %4 = fadd float %0, %1
  %5 = fsub float %4, %2
  %6 = fmul float %5, %3
  %7 = fdiv float %6, %4
  ret float %7
}

; CHECK-LABEL: tune_leon2:
; CHECK:       ld
; CHECK-NEXT:  ld
; CHECK-NEXT:  fadds
; CHECK-NEXT:  ld
; CHECK-NEXT:  fsubs
; CHECK-NEXT:  ld
; CHECK-NEXT:  fmuls
; CHECK-NEXT:  retl
; CHECK-NEXT:  fdivs

define float @tune_leon2(ptr byval(float) %a, ptr byval(float) %b, ptr byval(float) %c, ptr byval(float) %d) #0 {
entry:
  %0 = load float, ptr %a, align 8
  %1 = load float, ptr %b, align 8
  %2 = load float, ptr %c, align 8
  %3 = load float, ptr %d, align 8
  %4 = fadd float %0, %1
  %5 = fsub float %4, %2
  %6 = fmul float %5, %3
  %7 = fdiv float %6, %4
  ret float %7
}

; CHECK-LABEL: no_tune_after:
; CHECK:       ld
; CHECK-NEXT:  ld
; CHECK-NEXT:  ld
; CHECK-NEXT:  fadds
; CHECK-NEXT:  ld
; CHECK-NEXT:  fsubs
; CHECK-NEXT:  fmuls
; CHECK-NEXT:  retl
; CHECK-NEXT:  fdivs

define float @no_tune_after(ptr byval(float) %a, ptr byval(float) %b, ptr byval(float) %c, ptr byval(float) %d) {
entry:
  %0 = load float, ptr %a, align 8
  %1 = load float, ptr %b, align 8
  %2 = load float, ptr %c, align 8
  %3 = load float, ptr %d, align 8
  %4 = fadd float %0, %1
  %5 = fsub float %4, %2
  %6 = fmul float %5, %3
  %7 = fdiv float %6, %4
  ret float %7
}

attributes #0 = { "tune-cpu"="leon2" }
