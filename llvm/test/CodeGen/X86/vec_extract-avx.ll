target triple = "x86_64-unknown-unknown"

; RUN: llc < %s -march=x86-64 -mattr=+avx | FileCheck %s

; When extracting multiple consecutive elements from a larger
; vector into a smaller one, do it efficiently. We should use
; an EXTRACT_SUBVECTOR node internally rather than a bunch of
; single element extractions. 

; Extracting the low elements only requires using the right kind of store.
define void @low_v8f32_to_v4f32(<8 x float> %v, <4 x float>* %ptr) {
  %ext0 = extractelement <8 x float> %v, i32 0
  %ext1 = extractelement <8 x float> %v, i32 1
  %ext2 = extractelement <8 x float> %v, i32 2
  %ext3 = extractelement <8 x float> %v, i32 3
  %ins0 = insertelement <4 x float> undef, float %ext0, i32 0
  %ins1 = insertelement <4 x float> %ins0, float %ext1, i32 1
  %ins2 = insertelement <4 x float> %ins1, float %ext2, i32 2
  %ins3 = insertelement <4 x float> %ins2, float %ext3, i32 3
  store <4 x float> %ins3, <4 x float>* %ptr, align 16
  ret void

; CHECK-LABEL: low_v8f32_to_v4f32
; CHECK: vmovaps
; CHECK-NEXT: vzeroupper
; CHECK-NEXT: retq
}

; Extracting the high elements requires just one AVX instruction. 
define void @high_v8f32_to_v4f32(<8 x float> %v, <4 x float>* %ptr) {
  %ext0 = extractelement <8 x float> %v, i32 4
  %ext1 = extractelement <8 x float> %v, i32 5
  %ext2 = extractelement <8 x float> %v, i32 6
  %ext3 = extractelement <8 x float> %v, i32 7
  %ins0 = insertelement <4 x float> undef, float %ext0, i32 0
  %ins1 = insertelement <4 x float> %ins0, float %ext1, i32 1
  %ins2 = insertelement <4 x float> %ins1, float %ext2, i32 2
  %ins3 = insertelement <4 x float> %ins2, float %ext3, i32 3
  store <4 x float> %ins3, <4 x float>* %ptr, align 16
  ret void

; CHECK-LABEL: high_v8f32_to_v4f32
; CHECK: vextractf128
; CHECK-NEXT: vzeroupper
; CHECK-NEXT: retq
}

; Make sure element type doesn't alter the codegen. Note that
; if we were actually using the vector in this function and
; have AVX2, we should generate vextracti128 (the int version).
define void @high_v8i32_to_v4i32(<8 x i32> %v, <4 x i32>* %ptr) {
  %ext0 = extractelement <8 x i32> %v, i32 4
  %ext1 = extractelement <8 x i32> %v, i32 5
  %ext2 = extractelement <8 x i32> %v, i32 6
  %ext3 = extractelement <8 x i32> %v, i32 7
  %ins0 = insertelement <4 x i32> undef, i32 %ext0, i32 0
  %ins1 = insertelement <4 x i32> %ins0, i32 %ext1, i32 1
  %ins2 = insertelement <4 x i32> %ins1, i32 %ext2, i32 2
  %ins3 = insertelement <4 x i32> %ins2, i32 %ext3, i32 3
  store <4 x i32> %ins3, <4 x i32>* %ptr, align 16
  ret void

; CHECK-LABEL: high_v8i32_to_v4i32
; CHECK: vextractf128
; CHECK-NEXT: vzeroupper
; CHECK-NEXT: retq
}

; Make sure that element size doesn't alter the codegen.
define void @high_v4f64_to_v2f64(<4 x double> %v, <2 x double>* %ptr) {
  %ext0 = extractelement <4 x double> %v, i32 2
  %ext1 = extractelement <4 x double> %v, i32 3
  %ins0 = insertelement <2 x double> undef, double %ext0, i32 0
  %ins1 = insertelement <2 x double> %ins0, double %ext1, i32 1
  store <2 x double> %ins1, <2 x double>* %ptr, align 16
  ret void

; CHECK-LABEL: high_v4f64_to_v2f64
; CHECK: vextractf128
; CHECK-NEXT: vzeroupper
; CHECK-NEXT: retq
}
