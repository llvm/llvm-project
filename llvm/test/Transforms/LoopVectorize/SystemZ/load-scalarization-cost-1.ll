; RUN: opt -mtriple=s390x-unknown-linux -mcpu=z13 -loop-vectorize \
; RUN:   -force-vector-width=4 -debug-only=loop-vectorize \
; RUN:   -enable-interleaved-mem-accesses=false -disable-output < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Check that a scalarized load does not get a zero cost in a vectorized
; loop. It can only be folded into the add operand in the scalar loop.

define i32 @fun(i64* %data, i64 %n, i64 %s, i32* %Src) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %acc = phi i32 [ 0, %entry ], [ %acc_next, %for.body ]
  %gep = getelementptr inbounds i32, i32* %Src, i64 %iv
  %ld = load i32, i32* %gep
  %acc_next = add i32 %acc, %ld
  %iv.next = add nuw nsw i64 %iv, 2
  %cmp110.us = icmp slt i64 %iv.next, %n
  br i1 %cmp110.us, label %for.body, label %for.end

for.end:
  ret i32 %acc_next

; CHECK: Found an estimated cost of 4 for VF 4 For instruction:   %ld = load i32, i32* %gep
}
