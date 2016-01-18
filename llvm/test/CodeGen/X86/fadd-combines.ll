; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

define float @fadd_zero_f32(float %x) #0 {
; CHECK-LABEL: fadd_zero_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    retq
  %y = fadd float %x, 0.0
  ret float %y
}

define <4 x float> @fadd_zero_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_zero_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    retq
  %y = fadd <4 x float> %x, zeroinitializer
  ret <4 x float> %y
}

; CHECK: float 3
define float @fadd_2const_f32(float %x) #0 {
; CHECK-LABEL: fadd_2const_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    addss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd float %x, 1.0
  %z = fadd float %y, 2.0
  ret float %z
}

; CHECK: float 5.000000e+00
; CHECK: float 5.000000e+00
; CHECK: float 5.000000e+00
; CHECK: float 5.000000e+00
define <4 x float> @fadd_2const_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_2const_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    addps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %z = fadd <4 x float> %y, <float 4.0, float 3.0, float 2.0, float 1.0>
  ret <4 x float> %z
}

; CHECK: float 3
define float @fadd_x_fmul_x_c_f32(float %x) #0 {
; CHECK-LABEL: fadd_x_fmul_x_c_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fmul float %x, 2.0
  %z = fadd float %x, %y
  ret float %z
}

; CHECK: float 2.000000e+00
; CHECK: float 3.000000e+00
; CHECK: float 4.000000e+00
; CHECK: float 5.000000e+00
define <4 x float> @fadd_x_fmul_x_c_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_x_fmul_x_c_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fmul <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %z = fadd <4 x float> %x, %y
  ret <4 x float> %z
}

; CHECK: float 3
define float @fadd_fmul_x_c_x_f32(float %x) #0 {
; CHECK-LABEL: fadd_fmul_x_c_x_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fmul float %x, 2.0
  %z = fadd float %y, %x
  ret float %z
}

; CHECK: float 2.000000e+00
; CHECK: float 3.000000e+00
; CHECK: float 4.000000e+00
; CHECK: float 5.000000e+00
define <4 x float> @fadd_fmul_x_c_x_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_fmul_x_c_x_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fmul <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %z = fadd <4 x float> %y, %x
  ret <4 x float> %z
}

; CHECK: float 4
define float @fadd_fadd_x_x_fmul_x_c_f32(float %x) #0 {
; CHECK-LABEL: fadd_fadd_x_x_fmul_x_c_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd float %x, %x
  %z = fmul float %x, 2.0
  %w = fadd float %y, %z
  ret float %w
}

; CHECK: float 3.000000e+00
; CHECK: float 4.000000e+00
; CHECK: float 5.000000e+00
; CHECK: float 6.000000e+00
define <4 x float> @fadd_fadd_x_x_fmul_x_c_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_fadd_x_x_fmul_x_c_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd <4 x float> %x, %x
  %z = fmul <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %w = fadd <4 x float> %y, %z
  ret <4 x float> %w
}

; CHECK: float 4
define float @fadd_fmul_x_c_fadd_x_x_f32(float %x) #0 {
; CHECK-LABEL: fadd_fmul_x_c_fadd_x_x_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd float %x, %x
  %z = fmul float %x, 2.0
  %w = fadd float %z, %y
  ret float %w
}

; CHECK: float 3.000000e+00
; CHECK: float 4.000000e+00
; CHECK: float 5.000000e+00
; CHECK: float 6.000000e+00
define <4 x float> @fadd_fmul_x_c_fadd_x_x_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_fmul_x_c_fadd_x_x_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd <4 x float> %x, %x
  %z = fmul <4 x float> %x, <float 1.0, float 2.0, float 3.0, float 4.0>
  %w = fadd <4 x float> %z, %y
  ret <4 x float> %w
}

; CHECK: float 3
define float @fadd_x_fadd_x_x_f32(float %x) #0 {
; CHECK-LABEL: fadd_x_fadd_x_x_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd float %x, %x
  %z = fadd float %x, %y
  ret float %z
}

; CHECK: float 3.000000e+00
; CHECK: float 3.000000e+00
; CHECK: float 3.000000e+00
; CHECK: float 3.000000e+00
define <4 x float> @fadd_x_fadd_x_x_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_x_fadd_x_x_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd <4 x float> %x, %x
  %z = fadd <4 x float> %x, %y
  ret <4 x float> %z
}

; CHECK: float 3
define float @fadd_fadd_x_x_x_f32(float %x) #0 {
; CHECK-LABEL: fadd_fadd_x_x_x_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd float %x, %x
  %z = fadd float %y, %x
  ret float %z
}

; CHECK: float 3.000000e+00
; CHECK: float 3.000000e+00
; CHECK: float 3.000000e+00
; CHECK: float 3.000000e+00
define <4 x float> @fadd_fadd_x_x_x_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_fadd_x_x_x_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd <4 x float> %x, %x
  %z = fadd <4 x float> %y, %x
  ret <4 x float> %z
}

; CHECK: float 4
define float @fadd_fadd_x_x_fadd_x_x_f32(float %x) #0 {
; CHECK-LABEL: fadd_fadd_x_x_fadd_x_x_f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulss {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd float %x, %x
  %z = fadd float %y, %y
  ret float %z
}

; CHECK: float 4.000000e+00
; CHECK: float 4.000000e+00
; CHECK: float 4.000000e+00
; CHECK: float 4.000000e+00
define <4 x float> @fadd_fadd_x_x_fadd_x_x_4f32(<4 x float> %x) #0 {
; CHECK-LABEL: fadd_fadd_x_x_fadd_x_x_4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    mulps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %y = fadd <4 x float> %x, %x
  %z = fadd <4 x float> %y, %y
  ret <4 x float> %z
}

attributes #0 = { "less-precise-fpmad"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
