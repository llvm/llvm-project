; REQUIRES: asserts
; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b < %s -o /dev/null
;
; Verify that createHvxPrefixPred always stores 32-bit values in Words[].
; Commit e5bb946d1818f introduced a path where the i64 P2D result was pushed
; directly into Words[] without splitting into HiHalf/LoHalf, and the
; contraction loop operated on individual words instead of pairing them.
; This caused either a contractPredicate assertion (v2i1) or a BitTracker
; WD >= WS assertion (v8i1) when VINSERTW0 received an i64 operand.
;
; Regression test for https://github.com/llvm/llvm-project/issues/181362

; v2i1: Bytes=4, BitBytes=1 (needs two contraction steps)
define <128 x i1> @test_insert_v2i1(<128 x i1> %pred, <2 x i1> %sub) #0 {
  %res = call <128 x i1> @llvm.vector.insert.v128i1.v2i1(<128 x i1> %pred, <2 x i1> %sub, i64 0)
  ret <128 x i1> %res
}

; v4i1: Bytes=2, BitBytes=1 (needs one contraction step)
define <128 x i1> @test_insert_v4i1(<128 x i1> %pred, <4 x i1> %sub) #0 {
  %res = call <128 x i1> @llvm.vector.insert.v128i1.v4i1(<128 x i1> %pred, <4 x i1> %sub, i64 0)
  ret <128 x i1> %res
}

; v8i1: Bytes=1, BitBytes=1 (no contraction, but i64 was pushed into VINSERTW0)
define <128 x i1> @test_insert_v8i1(<128 x i1> %pred, <8 x i1> %sub) #0 {
  %res = call <128 x i1> @llvm.vector.insert.v128i1.v8i1(<128 x i1> %pred, <8 x i1> %sub, i64 0)
  ret <128 x i1> %res
}

declare <128 x i1> @llvm.vector.insert.v128i1.v2i1(<128 x i1>, <2 x i1>, i64)
declare <128 x i1> @llvm.vector.insert.v128i1.v4i1(<128 x i1>, <4 x i1>, i64)
declare <128 x i1> @llvm.vector.insert.v128i1.v8i1(<128 x i1>, <8 x i1>, i64)

attributes #0 = { nounwind "target-features"="+hvxv68,+hvx-length128b" }
