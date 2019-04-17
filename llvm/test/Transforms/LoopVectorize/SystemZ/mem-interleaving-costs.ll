; REQUIRES: asserts
; RUN: opt -mtriple=s390x-unknown-linux -mcpu=z13 -loop-vectorize \
; RUN:   -force-vector-width=4 -debug-only=loop-vectorize,vectorutils \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s
;
; Check that the loop vectorizer performs memory interleaving with accurate
; cost estimations.


; Simple case where just the load is interleaved, because the store group
; would have gaps.
define void @fun0(i32* %data, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds i32, i32* %data, i64 %i
  %tmp1 = load i32, i32* %tmp0, align 4
  %tmp2 = add i32 %tmp1, 1
  store i32 %tmp2, i32* %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void

; CHECK: LV: Creating an interleave group with:  %tmp1 = load i32, i32* %tmp0, align 4
; CHECK: LV: Found an estimated cost of 3 for VF 4 For instruction:   %tmp1 = load i32, i32* %tmp0, align 4
;        (vl; vl; vperm)
}

; Interleaving of both load and stores.
define void @fun1(i32* %data, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds i32, i32* %data, i64 %i
  %tmp1 = load i32, i32* %tmp0, align 4
  %i_1  = add i64 %i, 1
  %tmp2 = getelementptr inbounds i32, i32* %data, i64 %i_1
  %tmp3 = load i32, i32* %tmp2, align 4
  store i32 %tmp1, i32* %tmp2, align 4
  store i32 %tmp3, i32* %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void

; CHECK: LV: Creating an interleave group with:  store i32 %tmp3, i32* %tmp0, align 4
; CHECK: LV: Inserted:  store i32 %tmp1, i32* %tmp2, align 4
; CHECK:     into the interleave group with  store i32 %tmp3, i32* %tmp0, align 4
; CHECK: LV: Creating an interleave group with:  %tmp3 = load i32, i32* %tmp2, align 4
; CHECK: LV: Inserted:  %tmp1 = load i32, i32* %tmp0, align 4
; CHECK:     into the interleave group with  %tmp3 = load i32, i32* %tmp2, align 4

; CHECK: LV: Found an estimated cost of 4 for VF 4 For instruction:   %tmp1 = load i32, i32* %tmp0, align 4
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %tmp3 = load i32, i32* %tmp2, align 4
;            (vl; vl; vperm, vpkg)

; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   store i32 %tmp1, i32* %tmp2, align 4
; CHECK: LV: Found an estimated cost of 4 for VF 4 For instruction:   store i32 %tmp3, i32* %tmp0, align 4
;            (vmrlf; vmrhf; vst; vst)
}

