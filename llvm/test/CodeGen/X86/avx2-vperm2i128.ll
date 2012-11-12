; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vperm2i128 $17
define <32 x i8> @E(<32 x i8> %a, <32 x i8> %b) nounwind uwtable readnone ssp {
entry:
  ; add forces execution domain
  %a2 = add <32 x i8> %a, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %shuffle = shufflevector <32 x i8> %a2, <32 x i8> %b, <32 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <32 x i8> %shuffle
}

; CHECK: vperm2i128 $3
define <4 x i64> @E2(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; add forces execution domain
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %shuffle = shufflevector <4 x i64> %a2, <4 x i64> %b, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i64> %shuffle
}

; CHECK: vperm2i128 $49
define <8 x i32> @E3(<8 x i32> %a, <8 x i32> %b) nounwind uwtable readnone ssp {
entry:
  ; add forces execution domain
  %a2 = add <8 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %shuffle = shufflevector <8 x i32> %a2, <8 x i32> %b, <8 x i32> <i32 undef, i32 5, i32 undef, i32 7, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i32> %shuffle
}

; CHECK: vperm2i128 $2
define <16 x i16> @E4(<16 x i16> %a, <16 x i16> %b) nounwind uwtable readnone ssp {
entry:
  ; add forces execution domain
  %a2 = add <16 x i16> %a, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %shuffle = shufflevector <16 x i16> %a2, <16 x i16> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <16 x i16> %shuffle
}

; CHECK: vperm2i128 $2, (%
define <16 x i16> @E5(<16 x i16>* %a, <16 x i16>* %b) nounwind uwtable readnone ssp {
entry:
  %c = load <16 x i16>* %a
  %d = load <16 x i16>* %b
  %c2 = add <16 x i16> %c, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %shuffle = shufflevector <16 x i16> %c2, <16 x i16> %d, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <16 x i16> %shuffle
}
