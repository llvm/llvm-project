; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vpermilps
define <8 x float> @funcA(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> <i32 1, i32 2, i32 3, i32 1, i32 5, i32 6, i32 7, i32 5>
  ret <8 x float> %shuffle
}

; CHECK: vpermilpd
define <4 x double> @funcB(<4 x double> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 3>
  ret <4 x double> %shuffle
}

; CHECK: vpermilps
define <8 x i32> @funcC(<8 x i32> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> undef, <8 x i32> <i32 1, i32 2, i32 3, i32 1, i32 5, i32 6, i32 7, i32 5>
  ret <8 x i32> %shuffle
}

; CHECK: vpermilpd
define <4 x i64> @funcD(<4 x i64> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 3>
  ret <4 x i64> %shuffle
}

; CHECK: vpermilpd
define <4 x i64> @funcQ(<4 x i64>* %a) nounwind uwtable readnone ssp {
entry:
  %a2 = load <4 x i64>* %a
  %shuffle = shufflevector <4 x i64> %a2, <4 x i64> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 3>
  ret <4 x i64> %shuffle
}

; vpermil should match masks like this: <u,3,1,2,4,u,5,6>. Check that the
; target specific mask was correctly generated.
; CHECK: vpermilps $-100
define <8 x float> @funcE(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> <i32 8, i32 3, i32 1, i32 2, i32 4, i32 8, i32 5, i32 6>
  ret <8 x float> %shuffle
}

; CHECK: palignr
; CHECK: palignr
define <8 x float> @funcF(<8 x float> %a) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> zeroinitializer, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  ret <8 x float> %shuffle
}
