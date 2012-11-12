; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: _A
; CHECK: vperm2f128 $1
define <8 x float> @A(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <8 x float> %shuffle
}

; CHECK: _B
; CHECK: vperm2f128 $48
define <8 x float> @B(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15>
  ret <8 x float> %shuffle
}

; CHECK: _C
; CHECK: vperm2f128 $0
define <8 x float> @C(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  ret <8 x float> %shuffle
}

; CHECK: _D
; CHECK: vperm2f128 $17
define <8 x float> @D(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %shuffle
}

; CHECK: _E
; CHECK: vperm2f128 $17
define <32 x i8> @E(<32 x i8> %a, <32 x i8> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <32 x i8> %shuffle
}

; CHECK: _E2
; CHECK: vperm2f128 $3
define <4 x i64> @E2(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i64> %shuffle
}

;;;; Cases with undef indicies mixed in the mask

; CHECK: _F
; CHECK: vperm2f128 $33
define <8 x float> @F(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 undef, i32 undef, i32 6, i32 7, i32 undef, i32 9, i32 undef, i32 11>
  ret <8 x float> %shuffle
}

;;;; Cases we must not select vperm2f128

; CHECK: _G
; CHECK-NOT: vperm2f128
define <8 x float> @G(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 undef, i32 undef, i32 6, i32 7, i32 undef, i32 12, i32 undef, i32 15>
  ret <8 x float> %shuffle
}
