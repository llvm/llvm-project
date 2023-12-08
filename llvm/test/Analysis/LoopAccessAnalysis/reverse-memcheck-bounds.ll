; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

; The runtime memory check code and the access grouping
; algorithm both assume that the start and end values
; for an access range are ordered (start <= stop).
; When generating checks for accesses with negative stride
; we need to take this into account and swap the interval
; ends.
;
;   for (i = 0; i < 10000; i++) {
;     B[i] = A[15000 - i] * 3;
;   }

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnueabi"

; CHECK: function 'f':
; CHECK: (Low: (20000 + %a)<nuw> High: (60004 + %a))

@B = common global ptr null, align 8
@A = common global ptr null, align 8

define void @f() {
entry:
  %a = load ptr, ptr @A, align 8
  %b = load ptr, ptr @B, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %idx = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %negidx = sub i64 15000, %idx

  %arrayidxA0 = getelementptr inbounds i32, ptr %a, i64 %negidx
  %loadA0 = load i32, ptr %arrayidxA0, align 2

  %res = mul i32 %loadA0, 3

  %add = add nuw nsw i64 %idx, 1

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %idx
  store i32 %res, ptr %arrayidxB, align 2

  %exitcond = icmp eq i64 %idx, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK: function 'g':
; When the stride is not constant, we are forced to do umin/umax to get
; the interval limits.

;   for (i = 0; i < 10000; i++) {
;     B[i] = A[i] * 3;
;   }

; Here it is not obvious what the limits are, since 'step' could be negative.

; CHECK: Low: ((60000 + %a) umin (60000 + (-40000 * %step) + %a)) 
; CHECK: High: (4 + ((60000 + %a) umax (60000 + (-40000 * %step) + %a)))

define void @g(i64 %step) {
entry:
  %a = load ptr, ptr @A, align 8
  %b = load ptr, ptr @B, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %idx = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %idx_mul = mul i64 %idx, %step
  %negidx = sub i64 15000, %idx_mul

  %arrayidxA0 = getelementptr inbounds i32, ptr %a, i64 %negidx
  %loadA0 = load i32, ptr %arrayidxA0, align 2

  %res = mul i32 %loadA0, 3

  %add = add nuw nsw i64 %idx, 1

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %idx
  store i32 %res, ptr %arrayidxB, align 2

  %exitcond = icmp eq i64 %idx, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
