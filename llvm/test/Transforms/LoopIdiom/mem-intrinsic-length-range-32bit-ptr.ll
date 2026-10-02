; RUN: opt -passes=loop-idiom -S < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32-S64"

; void i64_iv_bounded(int *p, unsigned long long start) {
;   for (unsigned long long i = start; i < 4; ++i)
;     p[i] = -1;
; }

define void @i64_iv_bounded(ptr %p, i64 %start) {
; CHECK-LABEL: define void @i64_iv_bounded(
; CHECK:       call void @llvm.memset.p0.i32(ptr align 4 {{%.*}}, i8 -1, i32 range(i32 0, 17) {{%.*}}, i1 false)
entry:
  %guard = icmp ult i64 %start, 4
  br i1 %guard, label %loop, label %exit
loop:
  %i = phi i64 [ %start, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %i
  store i32 -1, ptr %gep, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, 4
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; void i64_iv_too_large(char *p, unsigned long long start) {
;   for (unsigned long long i = start; i < (1ULL << 40); ++i)
;     p[i] = 0;
; }

define void @i64_iv_too_large(ptr %p, i64 %start) {
; CHECK-LABEL: define void @i64_iv_too_large(
; CHECK:       call void @llvm.memset.p0.i32(ptr align 1 {{%.*}}, i8 0, i32 {{%.*}}, i1 false)
entry:
  %guard = icmp ult i64 %start, 1099511627776
  br i1 %guard, label %loop, label %exit
loop:
  %i = phi i64 [ %start, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i8, ptr %p, i64 %i
  store i8 0, ptr %gep, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, 1099511627776
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; An i64 IV runs up to 2^32 with no guard, so the max backedge-taken count,
; 2^32 - 1, still fits in the i32 length, but the max trip count doesn't.

define void @i64_iv_trip_count_too_large(ptr %p, i64 %start) {
; CHECK-LABEL: define void @i64_iv_trip_count_too_large(
; CHECK:       call void @llvm.memset.p0.i32(ptr align 1 {{%.*}}, i8 0, i32 {{%.*}}, i1 false)
entry:
  br label %loop
loop:
  %i = phi i64 [ %start, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i8, ptr %p, i64 %i
  store i8 0, ptr %gep, align 1
  %i.next = add nuw i64 %i, 1
  %cmp = icmp ult i64 %i.next, 4294967296
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}
