; RUN: opt < %s -passes='function(require<cycles>,loop-vectorize)' -disable-output

; LoopVectorize simplifies loops before processing them. If CycleAnalysis was
; cached before the pass, it must not be reused after loop simplification
; changes the CFG when BlockFrequencyInfo is requested lazily.

define i16 @f(i1 %c) {
entry:
  br label %header

latch:
  %iv.next = add i16 %iv, 1
  %done = icmp eq i16 %iv.next, 0
  br i1 %done, label %second, label %header

header:
  %iv = phi i16 [ 0, %entry ], [ %iv.next, %latch ]
  br i1 %c, label %trap, label %latch

trap:
  unreachable

second:
  br i1 false, label %second, label %second
}
