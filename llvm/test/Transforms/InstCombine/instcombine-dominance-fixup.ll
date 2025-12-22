; NOTE: This test ensures InstCombine preserves dominance even when it
; reorders shifts through SimplifyDemandedBits/log2 folding.
;
; RUN: opt -passes=instcombine,verify -disable-output %s

define i64 @f(i64 %0, i64 %1) {
entry:
  %2 = shl nuw i64 1, %1
  %3 = lshr exact i64 %2, 1
  %4 = shl nuw i64 %3, %1
  %5 = srem i64 %0, %4
  ret i64 %5
}
