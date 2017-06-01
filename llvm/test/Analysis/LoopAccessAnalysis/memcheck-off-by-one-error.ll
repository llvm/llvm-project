; RUN: opt -analyze --loop-accesses %s | FileCheck %s

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

;CHECK: Printing analysis 'Loop Access Analysis' for function 'fastCopy'
;CHECK: (Low: %op High: (32 + %op))
;CHECK: (Low: %src High: (32 + %src))

define void @fastCopy(i8* nocapture readonly %src, i8* nocapture %op) {
entry:
  br label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %len.addr.07 = phi i32 [ %sub, %while.body ], [ 32, %while.body.preheader ]
  %op.addr.06 = phi i8* [ %add.ptr1, %while.body ], [ %op, %while.body.preheader ]
  %src.addr.05 = phi i8* [ %add.ptr, %while.body ], [ %src, %while.body.preheader ]
  %0 = bitcast i8* %src.addr.05 to i64*
  %1 = load i64, i64* %0, align 8
  %2 = bitcast i8* %op.addr.06 to i64*
  store i64 %1, i64* %2, align 8
  %add.ptr = getelementptr inbounds i8, i8* %src.addr.05, i64 8
  %add.ptr1 = getelementptr inbounds i8, i8* %op.addr.06, i64 8
  %sub = add nsw i32 %len.addr.07, -8
  %cmp = icmp sgt i32 %len.addr.07, 8
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}
