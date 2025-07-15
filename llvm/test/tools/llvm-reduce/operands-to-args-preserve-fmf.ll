; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

; INTERESTING-LABEL: define float @callee(
; INTERESTING: fadd float
define float @callee(float %a) {
  %x = fadd float %a, 1.0
  ret float %x
}

; INTERESTING-LABEL: define float @caller(
; INTERESTING: load float

; REDUCED-LABEL: define float @caller(ptr %ptr, float %val, float %callee.ret1) {
; REDUCED: %callee.ret12 = call nnan nsz float @callee(float %val, float 0.000000e+00), !fpmath !0
define float @caller(ptr %ptr) {
  %val = load float, ptr %ptr
  %callee.ret = call nnan nsz float @callee(float %val), !fpmath !0
  ret float %callee.ret
}

; REDUCED: !0 = !{float 2.000000e+00}
!0 = !{float 2.0}
