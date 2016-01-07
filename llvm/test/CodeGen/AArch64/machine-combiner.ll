; RUN: llc -mtriple=aarch64-gnu-linux -mcpu=cortex-a57 -enable-unsafe-fp-math < %s | FileCheck %s 

; Verify that the first two adds are independent regardless of how the inputs are
; commuted. The destination registers are used as source registers for the third add.

define float @reassociate_adds1(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL:   reassociate_adds1:
; CHECK:         fadd  s0, s0, s1
; CHECK-NEXT:    fadd  s1, s2, s3
; CHECK-NEXT:    fadd  s0, s0, s1
; CHECK-NEXT:    ret
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds2(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL:   reassociate_adds2:
; CHECK:         fadd  s0, s0, s1
; CHECK-NEXT:    fadd  s1, s2, s3
; CHECK-NEXT:    fadd  s0, s0, s1
; CHECK-NEXT:    ret
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds3(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL:   reassociate_adds3:
; CHECK:         s0, s0, s1
; CHECK-NEXT:    s1, s2, s3
; CHECK-NEXT:    s0, s0, s1
; CHECK-NEXT:    ret
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %x3, %t1
  ret float %t2
}

define float @reassociate_adds4(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL:   reassociate_adds4:
; CHECK:         s0, s0, s1
; CHECK-NEXT:    s1, s2, s3
; CHECK-NEXT:    s0, s0, s1
; CHECK-NEXT:    ret
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %x3, %t1
  ret float %t2
}

; Verify that we reassociate some of these ops. The optimal balanced tree of adds is not
; produced because that would cost more compile time.

define float @reassociate_adds5(float %x0, float %x1, float %x2, float %x3, float %x4, float %x5, float %x6, float %x7) {
; CHECK-LABEL:   reassociate_adds5:
; CHECK:         fadd  s0, s0, s1
; CHECK-NEXT:    fadd  s1, s2, s3
; CHECK-NEXT:    fadd  s0, s0, s1
; CHECK-NEXT:    fadd  s1, s4, s5
; CHECK-NEXT:    fadd  s1, s1, s6
; CHECK-NEXT:    fadd  s0, s0, s1
; CHECK-NEXT:    fadd  s0, s0, s7
; CHECK-NEXT:    ret
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  %t3 = fadd float %t2, %x4
  %t4 = fadd float %t3, %x5
  %t5 = fadd float %t4, %x6
  %t6 = fadd float %t5, %x7
  ret float %t6
}

; Verify that we only need two associative operations to reassociate the operands.
; Also, we should reassociate such that the result of the high latency division
; is used by the final 'add' rather than reassociating the %x3 operand with the
; division. The latter reassociation would not improve anything.

define float @reassociate_adds6(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL:   reassociate_adds6:
; CHECK:         fdiv  s0, s0, s1
; CHECK-NEXT:    fadd  s1, s2, s3
; CHECK-NEXT:    fadd  s0, s0, s1
; CHECK-NEXT:    ret
  %t0 = fdiv float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %x3, %t1
  ret float %t2
}

; Verify that scalar single-precision multiplies are reassociated.

define float @reassociate_muls1(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL:   reassociate_muls1:
; CHECK:         fdiv  s0, s0, s1
; CHECK-NEXT:    fmul  s1, s2, s3
; CHECK-NEXT:    fmul  s0, s0, s1
; CHECK-NEXT:    ret
  %t0 = fdiv float %x0, %x1
  %t1 = fmul float %x2, %t0
  %t2 = fmul float %x3, %t1
  ret float %t2
}

; Verify that scalar double-precision adds are reassociated.

define double @reassociate_adds_double(double %x0, double %x1, double %x2, double %x3) {
; CHECK-LABEL:   reassociate_adds_double:
; CHECK:         fdiv  d0, d0, d1
; CHECK-NEXT:    fadd  d1, d2, d3
; CHECK-NEXT:    fadd  d0, d0, d1
; CHECK-NEXT:    ret
  %t0 = fdiv double %x0, %x1
  %t1 = fadd double %x2, %t0
  %t2 = fadd double %x3, %t1
  ret double %t2
}

; Verify that scalar double-precision multiplies are reassociated.

define double @reassociate_muls_double(double %x0, double %x1, double %x2, double %x3) {
; CHECK-LABEL:   reassociate_muls_double:
; CHECK:         fdiv  d0, d0, d1
; CHECK-NEXT:    fmul  d1, d2, d3
; CHECK-NEXT:    fmul  d0, d0, d1
; CHECK-NEXT:    ret
  %t0 = fdiv double %x0, %x1
  %t1 = fmul double %x2, %t0
  %t2 = fmul double %x3, %t1
  ret double %t2
}

; Verify that we reassociate vector instructions too.

define <4 x float> @vector_reassociate_adds1(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL:   vector_reassociate_adds1:
; CHECK:         fadd  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    fadd  v1.4s, v2.4s, v3.4s
; CHECK-NEXT:    fadd  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    ret
  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %t0, %x2
  %t2 = fadd <4 x float> %t1, %x3
  ret <4 x float> %t2
}

define <4 x float> @vector_reassociate_adds2(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL:   vector_reassociate_adds2:
; CHECK:         fadd  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    fadd  v1.4s, v2.4s, v3.4s
; CHECK-NEXT:    fadd  v0.4s, v0.4s, v1.4s
  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %x2, %t0
  %t2 = fadd <4 x float> %t1, %x3
  ret <4 x float> %t2
}

define <4 x float> @vector_reassociate_adds3(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL:   vector_reassociate_adds3:
; CHECK:         fadd  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    fadd  v1.4s, v2.4s, v3.4s
; CHECK-NEXT:    fadd  v0.4s, v0.4s, v1.4s
  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %t0, %x2
  %t2 = fadd <4 x float> %x3, %t1
  ret <4 x float> %t2
}

define <4 x float> @vector_reassociate_adds4(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL:   vector_reassociate_adds4:
; CHECK:         fadd  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    fadd  v1.4s, v2.4s, v3.4s
; CHECK-NEXT:    fadd  v0.4s, v0.4s, v1.4s
  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %x2, %t0
  %t2 = fadd <4 x float> %x3, %t1
  ret <4 x float> %t2
}
; Verify that 128-bit vector single-precision multiplies are reassociated.

define <4 x float> @reassociate_muls_v4f32(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL:   reassociate_muls_v4f32:
; CHECK:         fadd  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    fmul  v1.4s, v2.4s, v3.4s
; CHECK-NEXT:    fmul  v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    ret
  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fmul <4 x float> %x2, %t0
  %t2 = fmul <4 x float> %x3, %t1
  ret <4 x float> %t2
}

; Verify that 128-bit vector double-precision multiplies are reassociated.

define <2 x double> @reassociate_muls_v2f64(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, <2 x double> %x3) {
; CHECK-LABEL:   reassociate_muls_v2f64:
; CHECK:         fadd  v0.2d, v0.2d, v1.2d
; CHECK-NEXT:    fmul  v1.2d, v2.2d, v3.2d
; CHECK-NEXT:    fmul  v0.2d, v0.2d, v1.2d
; CHECK-NEXT:    ret
  %t0 = fadd <2 x double> %x0, %x1
  %t1 = fmul <2 x double> %x2, %t0
  %t2 = fmul <2 x double> %x3, %t1
  ret <2 x double> %t2
}

; PR25016: https://llvm.org/bugs/show_bug.cgi?id=25016
; Verify that reassociation is not happening needlessly or wrongly.

declare double @bar()

define double @reassociate_adds_from_calls() {
; CHECK-LABEL: reassociate_adds_from_calls:
; CHECK:       bl   bar
; CHECK-NEXT:  mov  v8.16b, v0.16b 
; CHECK-NEXT:  bl   bar
; CHECK-NEXT:  mov  v9.16b, v0.16b
; CHECK-NEXT:  bl   bar
; CHECK-NEXT:  mov  v10.16b, v0.16b 
; CHECK-NEXT:  bl   bar
; CHECK:       fadd d1, d8, d9 
; CHECK-NEXT:  fadd d0, d10, d0
; CHECK-NEXT:  fadd d0, d1, d0
  %x0 = call double @bar()
  %x1 = call double @bar()
  %x2 = call double @bar()
  %x3 = call double @bar()
  %t0 = fadd double %x0, %x1
  %t1 = fadd double %t0, %x2
  %t2 = fadd double %t1, %x3
  ret double %t2
}

define double @already_reassociated() {
; CHECK-LABEL: already_reassociated:
; CHECK:       bl   bar
; CHECK-NEXT:  mov  v8.16b, v0.16b 
; CHECK-NEXT:  bl   bar
; CHECK-NEXT:  mov  v9.16b, v0.16b
; CHECK-NEXT:  bl   bar
; CHECK-NEXT:  mov  v10.16b, v0.16b 
; CHECK-NEXT:  bl   bar
; CHECK:       fadd d1, d8, d9 
; CHECK-NEXT:  fadd d0, d10, d0
; CHECK-NEXT:  fadd d0, d1, d0
  %x0 = call double @bar()
  %x1 = call double @bar()
  %x2 = call double @bar()
  %x3 = call double @bar()
  %t0 = fadd double %x0, %x1
  %t1 = fadd double %x2, %x3
  %t2 = fadd double %t0, %t1
  ret double %t2
}

