; RUN: llc < %s -mtriple=arm64-eabi

; The DAGCombiner tries to do following shrink:
;     Convert x+y to (VT)((SmallVT)x+(SmallVT)y)
; But currently it can't handle vector type and will trigger an assertion failure
; when it tries to generate an add mixed using vector type and scaler type.
; This test checks that such assertion failur should not happen.
define <1 x i64> @dotest(<1 x i64> %in0) {
entry:
  %0 = add <1 x i64> %in0, %in0
  %vshl_n = shl <1 x i64> %0, <i64 32>
  %vsra_n = ashr <1 x i64> %vshl_n, <i64 32>
  ret <1 x i64> %vsra_n
}
