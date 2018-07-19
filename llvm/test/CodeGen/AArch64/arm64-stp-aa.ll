; RUN: llc < %s -mtriple=arm64-eabi -enable-misched=false -aarch64-enable-stp-suppress=false -verify-machineinstrs | FileCheck %s

; The next set of tests makes sure we can combine the second instruction into
; the first.

; CHECK-LABEL: stp_int_aa
; CHECK: stp w0, w1, [x2]
; CHECK: ldr w0, [x2, #8]
; CHECK: ret
define i32 @stp_int_aa(i32 %a, i32 %b, i32* nocapture %p) nounwind {
  store i32 %a, i32* %p, align 4
  %ld.ptr = getelementptr inbounds i32, i32* %p, i64 2
  %tmp = load i32, i32* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  store i32 %b, i32* %add.ptr, align 4
  ret i32 %tmp
}

; CHECK-LABEL: stp_long_aa
; CHECK: stp x0, x1, [x2]
; CHECK: ldr x0, [x2, #16]
; CHECK: ret
define i64 @stp_long_aa(i64 %a, i64 %b, i64* nocapture %p) nounwind {
  store i64 %a, i64* %p, align 8
  %ld.ptr = getelementptr inbounds i64, i64* %p, i64 2
  %tmp = load i64, i64* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 1
  store i64 %b, i64* %add.ptr, align 8
  ret i64 %tmp
}

; CHECK-LABEL: stp_float_aa
; CHECK: stp s0, s1, [x0]
; CHECK: ldr s0, [x0, #8]
; CHECK: ret
define float @stp_float_aa(float %a, float %b, float* nocapture %p) nounwind {
  store float %a, float* %p, align 4
  %ld.ptr = getelementptr inbounds float, float* %p, i64 2
  %tmp = load float, float* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds float, float* %p, i64 1
  store float %b, float* %add.ptr, align 4
  ret float %tmp
}

; CHECK-LABEL: stp_double_aa
; CHECK: stp d0, d1, [x0]
; CHECK: ldr d0, [x0, #16]
; CHECK: ret
define double @stp_double_aa(double %a, double %b, double* nocapture %p) nounwind {
  store double %a, double* %p, align 8
  %ld.ptr = getelementptr inbounds double, double* %p, i64 2
  %tmp = load double, double* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds double, double* %p, i64 1
  store double %b, double* %add.ptr, align 8
  ret double %tmp
}

; The next set of tests makes sure we can combine the first instruction into
; the second.

; CHECK-LABEL: stp_int_aa_after
; CHECK: ldr w0, [x3, #4]
; CHECK: stp w1, w2, [x3]
; CHECK: ret
define i32 @stp_int_aa_after(i32 %w0, i32 %a, i32 %b, i32* nocapture %p) nounwind {
  store i32 %a, i32* %p, align 4
  %ld.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp = load i32, i32* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  store i32 %b, i32* %add.ptr, align 4
  ret i32 %tmp
}

; CHECK-LABEL: stp_long_aa_after
; CHECK: ldr x0, [x3, #8]
; CHECK: stp x1, x2, [x3]
; CHECK: ret
define i64 @stp_long_aa_after(i64 %x0, i64 %a, i64 %b, i64* nocapture %p) nounwind {
  store i64 %a, i64* %p, align 8
  %ld.ptr = getelementptr inbounds i64, i64* %p, i64 1
  %tmp = load i64, i64* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 1
  store i64 %b, i64* %add.ptr, align 8
  ret i64 %tmp
}

; CHECK-LABEL: stp_float_aa_after
; CHECK: ldr s0, [x0, #4]
; CHECK: stp s1, s2, [x0]
; CHECK: ret
define float @stp_float_aa_after(float %s0, float %a, float %b, float* nocapture %p) nounwind {
  store float %a, float* %p, align 4
  %ld.ptr = getelementptr inbounds float, float* %p, i64 1
  %tmp = load float, float* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds float, float* %p, i64 1
  store float %b, float* %add.ptr, align 4
  ret float %tmp
}

; CHECK-LABEL: stp_double_aa_after
; CHECK: ldr d0, [x0, #8]
; CHECK: stp d1, d2, [x0]
; CHECK: ret
define double @stp_double_aa_after(double %d0, double %a, double %b, double* nocapture %p) nounwind {
  store double %a, double* %p, align 8
  %ld.ptr = getelementptr inbounds double, double* %p, i64 1
  %tmp = load double, double* %ld.ptr, align 4
  %add.ptr = getelementptr inbounds double, double* %p, i64 1
  store double %b, double* %add.ptr, align 8
  ret double %tmp
}

; Check that the stores %c and %d are paired after the fadd instruction,
; and then the stores %a and %d are paired after proving that they do not
; depend on the the (%c, %d) pair.
;
; CHECK-LABEL: st1:
; CHECK: stp q0, q1, [x{{[0-9]+}}]
; CHECK: fadd
; CHECK: stp q2, q0, [x{{[0-9]+}}, #32]
define void @st1(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, float* %base, i64 %index) {
entry:
  %a0 = getelementptr inbounds float, float* %base, i64 %index
  %b0 = getelementptr float, float* %a0, i64 4
  %c0 = getelementptr float, float* %a0, i64 8
  %d0 = getelementptr float, float* %a0, i64 12

  %a1 = bitcast float* %a0 to <4 x float>*
  %b1 = bitcast float* %b0 to <4 x float>*
  %c1 = bitcast float* %c0 to <4 x float>*
  %d1 = bitcast float* %d0 to <4 x float>*

  store <4 x float> %c, <4 x float> * %c1, align 4
  store <4 x float> %a, <4 x float> * %a1, align 4

  ; This fadd forces the compiler to pair %c and %e after fadd, and leave the
  ; stores %a and %b separated by a stp. The dependence analysis needs then to
  ; prove that it is safe to move %b past the stp to be paired with %a.
  %e = fadd fast <4 x float> %d, %a

  store <4 x float> %e, <4 x float>* %d1, align 4
  store <4 x float> %b, <4 x float>* %b1, align 4

  ret void
}
