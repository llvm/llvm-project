; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -S -debug-only=loop-vectorize 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: all_scalar
; CHECK:       LV: Found scalar instruction: %i.next = add nuw nsw i64 %i, 2
; CHECK:       LV: Found an estimated cost of 1 for VF 2 For instruction: %i.next = add nuw nsw i64 %i, 2
; CHECK:       LV: Not considering vector loop of width 2 because it will not generate any vector instructions
;
define void @all_scalar(i64* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr i64, i64* %a, i64 %i
  store i64 0, i64* %tmp0, align 1
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: PR33193
; CHECK:       LV: Found scalar instruction: %i.next = zext i32 %j.next to i64
; CHECK:       LV: Found an estimated cost of 0 for VF 8 For instruction: %i.next = zext i32 %j.next to i64
; CHECK:       LV: Not considering vector loop of width 8 because it will not generate any vector instructions
%struct.a = type { i32, i8 }
define void @PR33193(%struct.a* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %j = phi i32 [ 0, %entry ], [ %j.next, %for.body ]
  %tmp0 = getelementptr inbounds %struct.a, %struct.a* %a, i64 %i, i32 1
  store i8 0, i8* %tmp0, align 4
  %j.next = add i32 %j, 1
  %i.next = zext i32 %j.next to i64
  %cond = icmp ugt i64 %n, %i.next
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
