; REQUIRES: asserts
; RUN: opt -mtriple=s390x-unknown-linux -mcpu=z13 -loop-vectorize \
; RUN:   -force-vector-width=2 -debug-only=loop-vectorize \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s

; Check costs for branches inside a vectorized loop around predicated
; blocks. Each such branch will be guarded with an extractelement from the
; vector compare plus a test under mask instruction. This cost is modelled on
; the extractelement of i1.

define void @fun(i32* %arr, i64 %trip.count) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %indvars.iv
  %l = load i32, i32* %arrayidx, align 4
  %cmp55 = icmp sgt i32 %l, 0
  br i1 %cmp55, label %if.then, label %for.inc

if.then:
  %sub = sub nsw i32 0, %l
  store i32 %sub, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  ret void

; CHECK: LV: Found an estimated cost of 7 for VF 2 For instruction:   br i1 %cmp55, label %if.then, label %for.inc
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction:   br label %for.inc
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   br i1 %exitcond, label %for.end.loopexit, label %for.body
}
