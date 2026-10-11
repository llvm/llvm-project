; RUN: opt < %s -passes='require<cycles>,loop-vectorize,print<block-freq>' -disable-output 2>&1 | FileCheck %s

;; Loop simplification changes the CFG before LoopVectorize lazily requests
;; BlockFrequencyInfo, so the CycleInfo cached by require<cycles> must not be
;; reused. %second.preheader and %second.backedge exist only after
;; simplification, so the frequencies below describe the simplified CFG.

; CHECK-LABEL: block-frequency-info: f
; CHECK-NEXT:  - entry: float = 1.0,
; CHECK-NEXT:  - latch: float = 32.0,
; CHECK-NEXT:  - second.preheader: float = 1.0,
; CHECK-NEXT:  - header: float = 32.0,
; CHECK-NEXT:  - trap: float = 0.000000014901,
; CHECK-NEXT:  - second: float = 4096.0,
; CHECK-NEXT:  - second.backedge: float = 4096.0,

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
