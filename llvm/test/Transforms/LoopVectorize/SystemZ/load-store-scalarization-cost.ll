; REQUIRES: asserts
; RUN: opt -mtriple=s390x-unknown-linux -mcpu=z13 -loop-vectorize \
; RUN:   -force-vector-width=4 -debug-only=loop-vectorize \
; RUN:   -disable-output -enable-interleaved-mem-accesses=false < %s 2>&1 | \
; RUN:   FileCheck %s
;
; Check that a scalarized load/store does not get a cost for insterts/
; extracts, since z13 supports element load/store.

define void @fun(i32* %data, i64 %n) {
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

; CHECK: LV: Scalarizing:  %tmp1 = load i32, i32* %tmp0, align 4
; CHECK: LV: Scalarizing:  store i32 %tmp2, i32* %tmp0, align 4

; CHECK: LV: Found an estimated cost of 4 for VF 4 For instruction:   %tmp1 = load i32, i32* %tmp0, align 4
; CHECK: LV: Found an estimated cost of 4 for VF 4 For instruction:   store i32 %tmp2, i32* %tmp0, align 4
}

