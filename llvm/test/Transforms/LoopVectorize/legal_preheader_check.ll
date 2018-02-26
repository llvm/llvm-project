; RUN: opt < %s -loop-vectorize -debug -S -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

; D40973
; Make sure LV legal bails out when the loop doesn't have a legal pre-header.

; CHECK: LV: Loop doesn't have a legal pre-header.

define void @inc(i32 %n, i8* %P) {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %BB1, label %BB2

BB1:
  indirectbr i8* %P, [label %.lr.ph]

BB2:
  br label %.lr.ph

.lr.ph:
  %indvars.iv = phi i32 [ %indvars.iv.next, %.lr.ph ], [ 0, %BB1 ], [ 0, %BB2 ]
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:
  ret void
}
