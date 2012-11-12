; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; RUN: llc -O0 < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s -check-prefix=CHECK_O0

; CHECK: vmovaps
; CHECK: vmovaps
; CHECK: vmovaps
; CHECK: vmovaps
; CHECK: vmovaps
; CHECK: vmovaps
define void @test_256_load(double* nocapture %d, float* nocapture %f, <4 x i64>* nocapture %i) nounwind uwtable ssp {
entry:
  %0 = bitcast double* %d to <4 x double>*
  %tmp1.i = load <4 x double>* %0, align 32
  %1 = bitcast float* %f to <8 x float>*
  %tmp1.i17 = load <8 x float>* %1, align 32
  %tmp1.i16 = load <4 x i64>* %i, align 32
  tail call void @dummy(<4 x double> %tmp1.i, <8 x float> %tmp1.i17, <4 x i64> %tmp1.i16) nounwind
  store <4 x double> %tmp1.i, <4 x double>* %0, align 32
  store <8 x float> %tmp1.i17, <8 x float>* %1, align 32
  store <4 x i64> %tmp1.i16, <4 x i64>* %i, align 32
  ret void
}

declare void @dummy(<4 x double>, <8 x float>, <4 x i64>)

;;
;; The two tests below check that we must fold load + scalar_to_vector
;; + ins_subvec+ zext into only a single vmovss or vmovsd or vinsertps from memory

; CHECK: mov00
define <8 x float> @mov00(<8 x float> %v, float * %ptr) nounwind {
  %val = load float* %ptr
; CHECK: vinsertps
; CHECK: vinsertf128
  %i0 = insertelement <8 x float> zeroinitializer, float %val, i32 0
  ret <8 x float> %i0
; CHECK: ret
}

; CHECK: mov01
define <4 x double> @mov01(<4 x double> %v, double * %ptr) nounwind {
  %val = load double* %ptr
; CHECK: vmovlpd
; CHECK: vinsertf128
  %i0 = insertelement <4 x double> zeroinitializer, double %val, i32 0
  ret <4 x double> %i0
; CHECK: ret
}

; CHECK: vmovaps  %ymm
define void @storev16i16(<16 x i16> %a) nounwind {
  store <16 x i16> %a, <16 x i16>* undef, align 32
  unreachable
}

; CHECK: vmovups  %ymm
define void @storev16i16_01(<16 x i16> %a) nounwind {
  store <16 x i16> %a, <16 x i16>* undef, align 4
  unreachable
}

; CHECK: vmovaps  %ymm
define void @storev32i8(<32 x i8> %a) nounwind {
  store <32 x i8> %a, <32 x i8>* undef, align 32
  unreachable
}

; CHECK: vmovups  %ymm
define void @storev32i8_01(<32 x i8> %a) nounwind {
  store <32 x i8> %a, <32 x i8>* undef, align 4
  unreachable
}

; It is faster to make two saves, if the data is already in XMM registers. For
; example, after making an integer operation.
; CHECK: _double_save
; CHECK-NOT: vinsertf128 $1
; CHECK-NOT: vinsertf128 $0
; CHECK: vmovaps %xmm
; CHECK: vmovaps %xmm
define void @double_save(<4 x i32> %A, <4 x i32> %B, <8 x i32>* %P) nounwind ssp {
entry:
  %Z = shufflevector <4 x i32>%A, <4 x i32>%B, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %Z, <8 x i32>* %P, align 16
  ret void
}

declare void @llvm.x86.avx.maskstore.ps.256(i8*, <8 x float>, <8 x float>) nounwind

; CHECK_O0: _f_f
; CHECK-O0: vmovss LCPI
; CHECK-O0: vxorps  %xmm
; CHECK-O0: vmovss %xmm
define void @f_f() nounwind {
allocas:
  br i1 undef, label %cif_mask_all, label %cif_mask_mixed

cif_mask_all:                                     ; preds = %allocas
  unreachable

cif_mask_mixed:                                   ; preds = %allocas
  br i1 undef, label %cif_mixed_test_all, label %cif_mixed_test_any_check

cif_mixed_test_all:                               ; preds = %cif_mask_mixed
  call void @llvm.x86.avx.maskstore.ps.256(i8* undef, <8 x float> <float 0xFFFFFFFFE0000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, <8 x float> undef) nounwind
  unreachable

cif_mixed_test_any_check:                         ; preds = %cif_mask_mixed
  unreachable
}

