; RUN: opt -passes='print<access-info>' %s -disable-output 2>&1 | FileCheck %s

; This test verifies run-time boundary check of memory accesses.
; The original loop:
;   void fastCopy(const char* src, char* op) {
;     int len = 32;
;     while (len > 0) {
;       *(reinterpret_cast<long long*>(op)) = *(reinterpret_cast<const long long*>(src));
;       src += 8;
;       op += 8;
;       len -= 8;
;     }
;   }
; Boundaries calculations before this patch:
; (Low: %src High: (24 + %src))
; and the actual distance between two pointers was 31,  (%op - %src = 31)
; IsConflict = (24 > 31) = false -> execution is directed to the vectorized loop.
; The loop was vectorized to 4, 32 byte memory access ( <4 x i64> ),
; store a value at *%op touched memory under *%src.

;CHECK: function 'fastCopy':
;CHECK: (Low: %op High: (32 + %op))
;CHECK: (Low: %src High: (32 + %src))

define void @fastCopy(ptr nocapture readonly %src, ptr nocapture %op) {
entry:
  br label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %len.addr.07 = phi i32 [ %sub, %while.body ], [ 32, %while.body.preheader ]
  %op.addr.06 = phi ptr [ %add.ptr1, %while.body ], [ %op, %while.body.preheader ]
  %src.addr.05 = phi ptr [ %add.ptr, %while.body ], [ %src, %while.body.preheader ]
  %0 = load i64, ptr %src.addr.05, align 8
  store i64 %0, ptr %op.addr.06, align 8
  %add.ptr = getelementptr inbounds i8, ptr %src.addr.05, i64 8
  %add.ptr1 = getelementptr inbounds i8, ptr %op.addr.06, i64 8
  %sub = add nsw i32 %len.addr.07, -8
  %cmp = icmp sgt i32 %len.addr.07, 8
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}
