; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s


; CHECK: vpunpcklbw %xmm
; CHECK-NEXT: vpunpckhbw %xmm
; CHECK-NEXT: vinsertf128 $1
; CHECK-NEXT: vpermilps $85
define <32 x i8> @funcA(<32 x i8> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> undef, <32 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <32 x i8> %shuffle
}

; CHECK: vpunpckhwd %xmm
; CHECK-NEXT: vinsertf128 $1
; CHECK-NEXT: vpermilps $85
define <16 x i16> @funcB(<16 x i16> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> undef, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <16 x i16> %shuffle
}

; CHECK: vmovd
; CHECK-NEXT: vmovlhps %xmm
; CHECK-NEXT: vinsertf128 $1
define <4 x i64> @funcC(i64 %q) nounwind uwtable readnone ssp {
entry:
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %vecinit2.i = insertelement <4 x i64> %vecinit.i, i64 %q, i32 1
  %vecinit4.i = insertelement <4 x i64> %vecinit2.i, i64 %q, i32 2
  %vecinit6.i = insertelement <4 x i64> %vecinit4.i, i64 %q, i32 3
  ret <4 x i64> %vecinit6.i
}

; CHECK: vpermilpd $0
; CHECK-NEXT: vinsertf128 $1
define <4 x double> @funcD(double %q) nounwind uwtable readnone ssp {
entry:
  %vecinit.i = insertelement <4 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <4 x double> %vecinit.i, double %q, i32 1
  %vecinit4.i = insertelement <4 x double> %vecinit2.i, double %q, i32 2
  %vecinit6.i = insertelement <4 x double> %vecinit4.i, double %q, i32 3
  ret <4 x double> %vecinit6.i
}

; Test this simple opt:
;   shuffle (scalar_to_vector (load (ptr + 4))), undef, <0, 0, 0, 0>
; To:
;   shuffle (vload ptr)), undef, <1, 1, 1, 1>
; CHECK: vmovaps
; CHECK-NEXT: vinsertf128  $1
; CHECK-NEXT: vpermilps $-1
define <8 x float> @funcE() nounwind {
allocas:
  %udx495 = alloca [18 x [18 x float]], align 32
  br label %for_test505.preheader

for_test505.preheader:                            ; preds = %for_test505.preheader, %allocas
  br i1 undef, label %for_exit499, label %for_test505.preheader

for_exit499:                                      ; preds = %for_test505.preheader
  br i1 undef, label %__load_and_broadcast_32.exit1249, label %load.i1247

load.i1247:                                       ; preds = %for_exit499
  %ptr1227 = getelementptr [18 x [18 x float]]* %udx495, i64 0, i64 1, i64 1
  %ptr.i1237 = bitcast float* %ptr1227 to i32*
  %val.i1238 = load i32* %ptr.i1237, align 4
  %ret6.i1245 = insertelement <8 x i32> undef, i32 %val.i1238, i32 6
  %ret7.i1246 = insertelement <8 x i32> %ret6.i1245, i32 %val.i1238, i32 7
  %phitmp = bitcast <8 x i32> %ret7.i1246 to <8 x float>
  br label %__load_and_broadcast_32.exit1249

__load_and_broadcast_32.exit1249:                 ; preds = %load.i1247, %for_exit499
  %load_broadcast12281250 = phi <8 x float> [ %phitmp, %load.i1247 ], [ undef, %for_exit499 ]
  ret <8 x float> %load_broadcast12281250
}

; CHECK: vinsertf128 $1
; CHECK-NEXT: vpermilps $0
define <8 x float> @funcF(i32 %val) nounwind {
  %ret6 = insertelement <8 x i32> undef, i32 %val, i32 6
  %ret7 = insertelement <8 x i32> %ret6, i32 %val, i32 7
  %tmp = bitcast <8 x i32> %ret7 to <8 x float>
  ret <8 x float> %tmp
}

; CHECK: vinsertf128  $1
; CHECK-NEXT: vpermilps  $0
define <8 x float> @funcG(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

; CHECK: vextractf128  $1
; CHECK-NEXT: vinsertf128  $1
; CHECK-NEXT: vpermilps  $85
define <8 x float> @funcH(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <8 x float> %shuffle
}

