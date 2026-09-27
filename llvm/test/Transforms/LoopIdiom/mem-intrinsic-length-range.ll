; RUN: opt -passes=loop-idiom -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; void bounded_by_exit_condition(int *p, unsigned long start) {
;   for (unsigned long i = start; i < 4; ++i)
;     p[i] = -1;
; }

define void @bounded_by_exit_condition(ptr %p, i64 %start) {
; CHECK-LABEL: define void @bounded_by_exit_condition(
; CHECK:       call void @llvm.memset.p0.i64(ptr align 4 {{%.*}}, i8 -1, i64 range(i64 0, 17) {{%.*}}, i1 false)
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

; void bounded_by_guard(char *p, unsigned long n) {
;   if (n <= 100 && n != 0)
;     for (unsigned long i = 0; i < n; ++i)
;       p[i] = 0;
; }

define void @bounded_by_guard(ptr %p, i64 %n) {
; CHECK-LABEL: define void @bounded_by_guard(
; CHECK:       call void @llvm.memset.p0.i64(ptr align 1 {{%.*}}, i8 0, i64 range(i64 0, 101) {{%.*}}, i1 false)
entry:
  %small = icmp ule i64 %n, 100
  %nonzero = icmp ne i64 %n, 0
  %guard = and i1 %small, %nonzero
  br i1 %guard, label %loop, label %exit
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i8, ptr %p, i64 %i
  store i8 0, ptr %gep, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; void unbounded(int *p, unsigned long n) {
;   if (n != 0)
;     for (unsigned long i = 0; i < n; ++i)
;       p[i] = -1;
; }

define void @unbounded(ptr %p, i64 %n) {
; CHECK-LABEL: define void @unbounded(
; CHECK:       call void @llvm.memset.p0.i64(ptr align 4 {{%.*}}, i8 -1, i64 {{%.*}}, i1 false)
entry:
  %nonzero = icmp ne i64 %n, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %i
  store i32 -1, ptr %gep, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}
