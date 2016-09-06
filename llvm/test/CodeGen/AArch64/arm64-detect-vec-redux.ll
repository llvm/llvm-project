; RUN: llc -mtriple=arm64-darwin-unknown < %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; Function Attrs: nounwind readnone
define i32 @dotests_56() #0 {
entry:
  %vqshrn_n4 = tail call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64> zeroinitializer, i32 19)
  %shuffle.i109 = shufflevector <2 x i32> %vqshrn_n4, <2 x i32> undef, <4 x i32> <i32 undef, i32 1, i32 undef, i32 undef>
  %neg = xor <4 x i32> %shuffle.i109, <i32 undef, i32 -1, i32 undef, i32 undef>
  %shuffle = shufflevector <4 x i32> %neg, <4 x i32> undef, <2 x i32> <i32 1, i32 undef>
  %mul = mul <2 x i32> %shuffle, <i32 add (i32 extractelement (<2 x i32> bitcast (<1 x i64> <i64 -4264345899313889281> to <2 x i32>), i32 0), i32 sub (i32 0, i32 extractelement (<2 x i32> bitcast (<1 x i64> <i64 -9223231295071453185> to <2 x i32>), i32 0))), i32 undef>
  %shuffle27 = shufflevector <2 x i32> %mul, <2 x i32> undef, <4 x i32> zeroinitializer
  %0 = bitcast <4 x i32> %shuffle27 to <8 x i16>
  %shuffle.i108 = shufflevector <8 x i16> %0, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  %vqshrn_n38 = tail call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> %shuffle.i108, i32 1)
  %shuffle.i = shufflevector <8 x i8> %vqshrn_n38, <8 x i8> undef, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1 = bitcast <16 x i8> %shuffle.i to <2 x i64>
  %vpaddq_v2.i = tail call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> undef, <2 x i64> %1) #2
  %vqdmlal2.i = tail call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> undef, <2 x i32> undef) #2
  %vqdmlal_v3.i = tail call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %vpaddq_v2.i, <2 x i64> %vqdmlal2.i) #2
  %vmovn.i = trunc <2 x i64> %vqdmlal_v3.i to <2 x i32>
  %vqdmulh_v2.i = tail call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %vmovn.i, <2 x i32> zeroinitializer) #2
  %2 = bitcast <2 x i32> %vqdmulh_v2.i to <1 x i64>
  %vget_lane = extractelement <1 x i64> %2, i32 0
  %cmp = icmp ne i64 %vget_lane, -7395147708962464393
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Function Attrs: nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64>, i32) #1

; Function Attrs: nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16>, i32) #1

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32>, <2 x i32>) #1

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64>, <2 x i64>) #1

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64>, <2 x i64>) #1

; Function Attrs: nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32>, <2 x i32>) #1

attributes #0 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
