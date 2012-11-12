; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vshufps  $-53, %ymm
define <8 x float> @A(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  ret <8 x float> %shuffle
}

; CHECK: vshufps  $-53, (%{{.*}}), %ymm
define <8 x float> @A2(<8 x float>* %a, <8 x float>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <8 x float>* %a
  %b2 = load <8 x float>* %b
  %shuffle = shufflevector <8 x float> %a2, <8 x float> %b2, <8 x i32> <i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  ret <8 x float> %shuffle
}

; CHECK: vshufps  $-53, %ymm
define <8 x i32> @A3(<8 x i32> %a, <8 x i32> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  ret <8 x i32> %shuffle
}

; CHECK: vshufps  $-53, (%{{.*}}), %ymm
define <8 x i32> @A4(<8 x i32>* %a, <8 x i32>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <8 x i32>* %a
  %b2 = load <8 x i32>* %b
  %shuffle = shufflevector <8 x i32> %a2, <8 x i32> %b2, <8 x i32> <i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  ret <8 x i32> %shuffle
}

; CHECK: vshufpd  $10, %ymm
define <4 x double> @B(<4 x double> %a, <4 x double> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x double> %shuffle
}

; CHECK: vshufpd  $10, (%{{.*}}), %ymm
define <4 x double> @B2(<4 x double>* %a, <4 x double>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <4 x double>* %a
  %b2 = load <4 x double>* %b
  %shuffle = shufflevector <4 x double> %a2, <4 x double> %b2, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x double> %shuffle
}

; CHECK: vshufpd  $10, %ymm
define <4 x i64> @B3(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i64> %shuffle
}

; CHECK: vshufpd  $10, (%{{.*}}), %ymm
define <4 x i64> @B4(<4 x i64>* %a, <4 x i64>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <4 x i64>* %a
  %b2 = load <4 x i64>* %b
  %shuffle = shufflevector <4 x i64> %a2, <4 x i64> %b2, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i64> %shuffle
}

; CHECK: vshufps  $-53, %ymm
define <8 x float> @C(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 undef, i32 undef, i32 11, i32 undef, i32 6, i32 12, i32 undef>
  ret <8 x float> %shuffle
}

; CHECK: vshufpd  $2, %ymm
define <4 x double> @D(<4 x double> %a, <4 x double> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 undef>
  ret <4 x double> %shuffle
}

; CHECK: vshufps $-55, %ymm
define <8 x float> @E(<8 x float> %a, <8 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 9, i32 10, i32 0, i32 3, i32 13, i32 14, i32 4, i32 7>
  ret <8 x float> %shuffle
}

; CHECK: vshufpd  $8, %ymm
define <4 x double> @F(<4 x double> %a, <4 x double> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 7>
  ret <4 x double> %shuffle
}

; CHECK: vshufps  $-53, %xmm
define <4 x float> @A128(<4 x float> %a, <4 x float> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 3, i32 2, i32 4, i32 7>
  ret <4 x float> %shuffle
}

; CHECK: vshufps  $-53, (%{{.*}}), %xmm
define <4 x float> @A2128(<4 x float>* %a, <4 x float>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <4 x float>* %a
  %b2 = load <4 x float>* %b
  %shuffle = shufflevector <4 x float> %a2, <4 x float> %b2, <4 x i32> <i32 3, i32 2, i32 4, i32 7>
  ret <4 x float> %shuffle
}

; CHECK: vshufps  $-53, %xmm
define <4 x i32> @A3128(<4 x i32> %a, <4 x i32> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 3, i32 2, i32 4, i32 7>
  ret <4 x i32> %shuffle
}

; CHECK: vshufps  $-53, (%{{.*}}), %xmm
define <4 x i32> @A4128(<4 x i32>* %a, <4 x i32>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <4 x i32>* %a
  %b2 = load <4 x i32>* %b
  %shuffle = shufflevector <4 x i32> %a2, <4 x i32> %b2, <4 x i32> <i32 3, i32 2, i32 4, i32 7>
  ret <4 x i32> %shuffle
}

; CHECK: vshufpd  $1, %xmm
define <2 x double> @B128(<2 x double> %a, <2 x double> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %shuffle
}

; CHECK: vshufpd  $1, (%{{.*}}), %xmm
define <2 x double> @B2128(<2 x double>* %a, <2 x double>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <2 x double>* %a
  %b2 = load <2 x double>* %b
  %shuffle = shufflevector <2 x double> %a2, <2 x double> %b2, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %shuffle
}

; CHECK: vshufpd  $1, %xmm
define <2 x i64> @B3128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
entry:
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}

; CHECK: vshufpd  $1, (%{{.*}}), %xmm
define <2 x i64> @B4128(<2 x i64>* %a, <2 x i64>* %b) nounwind uwtable readnone ssp {
entry:
  %a2 = load <2 x i64>* %a
  %b2 = load <2 x i64>* %b
  %shuffle = shufflevector <2 x i64> %a2, <2 x i64> %b2, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}
