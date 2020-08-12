; RUN: opt < %s -vector-combine -S -mtriple=x86_64-- -mattr=-sse | FileCheck %s --check-prefixes=CHECK

define <4 x float> @bitcast_shuf_same_size(<4 x i32> %v) {
; CHECK-LABEL: @bitcast_shuf_same_size(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[V:%.*]] to <4 x float>
; CHECK-NEXT:    [[R:%.*]] = shufflevector <4 x float> [[TMP1]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT:    ret <4 x float> [[R]]
;
  %shuf = shufflevector <4 x i32> %v, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %r = bitcast <4 x i32> %shuf to <4 x float>
  ret <4 x float> %r
}
