; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vxorps
; CHECK-NEXT: vinsertf128 $0
define <8 x float> @castA(<4 x float> %m) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x float> %m, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle.i
}

; CHECK: vxorps
; CHECK-NEXT: vinsertf128 $0
define <4 x double> @castB(<2 x double> %m) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <2 x double> %m, <2 x double> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  ret <4 x double> %shuffle.i
}

; CHECK: vxorps
; CHECK-NEXT: vinsertf128 $0
define <4 x i64> @castC(<2 x i64> %m) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <2 x i64> %m, <2 x i64> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  ret <4 x i64> %shuffle.i
}

; CHECK-NOT: vextractf128 $0
define <4 x float> @castD(<8 x float> %m) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %m, <8 x float> %m, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuffle.i
}

; CHECK-NOT: vextractf128 $0
define <2 x i64> @castE(<4 x i64> %m) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x i64> %m, <4 x i64> %m, <2 x i32> <i32 0, i32 1>
  ret <2 x i64> %shuffle.i
}

; CHECK-NOT: vextractf128 $0
define <2 x double> @castF(<4 x double> %m) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <4 x double> %m, <4 x double> %m, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuffle.i
}

