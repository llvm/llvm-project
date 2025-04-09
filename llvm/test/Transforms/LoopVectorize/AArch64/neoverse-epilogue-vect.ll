; RUN: opt -passes=loop-vectorize -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define noundef i32 @V1(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef readonly %1, i32 noundef %2) #0 {
; CHECK-LABEL: @V1(
; CHECK-NOT:   vec.epilog.ph:
; CHECK-NOT:   vec.epilog.vector.body:
; CHECK-NOT:   vec.epilog.middle.block:
; CHECK-NOT:   vec.epilog.scalar.ph:
;
entry:
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %8

5:
  %6 = zext nneg i32 %2 to i64
  br label %9

7:
  br label %8

8:
  ret i32 42

9:
  %10 = phi i64 [ 0, %5 ], [ %16, %9 ]
  %11 = getelementptr inbounds double, ptr %0, i64 %10
  %12 = load double, ptr %11, align 8
  %13 = getelementptr inbounds double, ptr %1, i64 %10
  %14 = load double, ptr %13, align 8
  %15 = fadd fast double %14, %12
  store double %15, ptr %11, align 8
  %16 = add nuw nsw i64 %10, 1
  %17 = icmp eq i64 %16, %6
  br i1 %17, label %7, label %9
}

define noundef i32 @V2(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef readonly %1, i32 noundef %2) #1 {
;
; CHECK-LABEL: @V2(
; CHECK:       vec.epilog.ph:
; CHECK:       vec.epilog.vector.body:
; CHECK:       vec.epilog.middle.block:
; CHECK:       vec.epilog.scalar.ph:
;
entry:
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %8

5:
  %6 = zext nneg i32 %2 to i64
  br label %9

7:
  br label %8

8:
  ret i32 42

9:
  %10 = phi i64 [ 0, %5 ], [ %16, %9 ]
  %11 = getelementptr inbounds double, ptr %0, i64 %10
  %12 = load double, ptr %11, align 8
  %13 = getelementptr inbounds double, ptr %1, i64 %10
  %14 = load double, ptr %13, align 8
  %15 = fadd fast double %14, %12
  store double %15, ptr %11, align 8
  %16 = add nuw nsw i64 %10, 1
  %17 = icmp eq i64 %16, %6
  br i1 %17, label %7, label %9
}

; TODO: The V3 will generate a scalable vector body, so doesn't need a
; epilogue loop, but will need to be checked that is really the best thing to
; for the V3.
;
define noundef i32 @V3(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef readonly %1, i32 noundef %2) #2 {
;
; CHECK-LABEL: @V3(
; CHECK-NOT:   vec.epilog.ph:
; CHECK-NOT:   vec.epilog.vector.body:
; CHECK-NOT:   vec.epilog.middle.block:
; CHECK-NOT:   vec.epilog.scalar.ph:
;
entry:
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %8

5:
  %6 = zext nneg i32 %2 to i64
  br label %9

7:
  br label %8

8:
  ret i32 42

9:
  %10 = phi i64 [ 0, %5 ], [ %16, %9 ]
  %11 = getelementptr inbounds double, ptr %0, i64 %10
  %12 = load double, ptr %11, align 8
  %13 = getelementptr inbounds double, ptr %1, i64 %10
  %14 = load double, ptr %13, align 8
  %15 = fadd fast double %14, %12
  store double %15, ptr %11, align 8
  %16 = add nuw nsw i64 %10, 1
  %17 = icmp eq i64 %16, %6
  br i1 %17, label %7, label %9
}

attributes #0 = { vscale_range(1,16) "target-cpu"="neoverse-v1" "target-features"="+sve2" }

attributes #1 = { vscale_range(1,16) "target-cpu"="neoverse-v2" "target-features"="+sve2" }

attributes #2 = { vscale_range(1,16) "target-cpu"="neoverse-v3" "target-features"="+sve2" }
