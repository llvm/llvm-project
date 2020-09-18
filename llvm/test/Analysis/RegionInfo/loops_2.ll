; REQUIRES: asserts
; RUN: opt -regions -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -regions -stats -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze -enable-new-pm=0 < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze -enable-new-pm=0 < %s 2>&1 | FileCheck -check-prefix=RNIT %s

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @meread_() nounwind {
entry:
  br label %bb23

bb23:
  br label %bb.i

bb.i:                                             ; preds = %bb.i, %bb54
  br label %pflini_.exit

pflini_.exit:                                     ; preds = %bb.i
  br label %bb58thread-split

bb58thread-split:                                 ; preds = %bb64, %bb61, %pflini_.exit
  br label %bb58

bb58:                                             ; preds = %bb60, %bb58thread-split
  br i1 1, label %bb59, label %bb23

bb59:                                             ; preds = %bb58
  switch i32 1, label %bb60 [
    i32 1, label %l98
  ]

bb60:                                             ; preds = %bb59
  br i1 1, label %bb61, label %bb58

bb61:                                             ; preds = %bb60
  br label %bb58thread-split

l98:                                   ; preds = %bb69, %bb59
  ret void
}
; CHECK-NOT: =>
; CHECK: [0] entry => <Function Return>
; CHECK: [1] bb23 => l98
; STAT: 2 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: entry, bb23, bb.i, pflini_.exit, bb58thread-split, bb58, bb59, bb60, bb61, l98,
; BBIT: bb23, bb.i, pflini_.exit, bb58thread-split, bb58, bb59, bb60, bb61,

; RNIT: entry, bb23 => l98, l98,
; RNIT: bb23, bb.i, pflini_.exit, bb58thread-split, bb58, bb59, bb60, bb61,
