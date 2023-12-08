; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s

; Make sure that we merge the consecutive load/store sequence below and use a
; word (16 bit) instead of a byte copy.
; CHECK: MergeLoadStoreBaseIndexOffset
; CHECK: ldrh    [[REG:r[0-9]+]], [{{.*}}]
; CHECK: strh    [[REG]], [r1], #2
define void @MergeLoadStoreBaseIndexOffset(ptr %a, ptr %b, ptr %c, i32 %n) {
  br label %1

; <label>:1
  %.09 = phi i32 [ %n, %0 ], [ %11, %1 ]
  %.08 = phi ptr [ %b, %0 ], [ %10, %1 ]
  %.0 = phi ptr [ %a, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i32, ptr %.0, i32 1
  %3 = load i32, ptr %.0, align 1
  %4 = getelementptr inbounds i8, ptr %c, i32 %3
  %5 = load i8, ptr %4, align 1
  %6 = add i32 %3, 1
  %7 = getelementptr inbounds i8, ptr %c, i32 %6
  %8 = load i8, ptr %7, align 1
  store i8 %5, ptr %.08, align 1
  %9 = getelementptr inbounds i8, ptr %.08, i32 1
  store i8 %8, ptr %9, align 1
  %10 = getelementptr inbounds i8, ptr %.08, i32 2
  %11 = add nsw i32 %.09, -1
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %1

; <label>:13
  ret void
}

; Make sure that we merge the consecutive load/store sequence below and use a
; word (16 bit) instead of a byte copy even if there are intermediate sign
; extensions.
; CHECK: MergeLoadStoreBaseIndexOffsetSext
; CHECK: ldrh    [[REG:r[0-9]+]], [{{.*}}]
; CHECK: strh    [[REG]], [r1], #2
define void @MergeLoadStoreBaseIndexOffsetSext(ptr %a, ptr %b, ptr %c, i32 %n) {
  br label %1

; <label>:1
  %.09 = phi i32 [ %n, %0 ], [ %12, %1 ]
  %.08 = phi ptr [ %b, %0 ], [ %11, %1 ]
  %.0 = phi ptr [ %a, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i8, ptr %.0, i32 1
  %3 = load i8, ptr %.0, align 1
  %4 = sext i8 %3 to i32
  %5 = getelementptr inbounds i8, ptr %c, i32 %4
  %6 = load i8, ptr %5, align 1
  %7 = add i32 %4, 1
  %8 = getelementptr inbounds i8, ptr %c, i32 %7
  %9 = load i8, ptr %8, align 1
  store i8 %6, ptr %.08, align 1
  %10 = getelementptr inbounds i8, ptr %.08, i32 1
  store i8 %9, ptr %10, align 1
  %11 = getelementptr inbounds i8, ptr %.08, i32 2
  %12 = add nsw i32 %.09, -1
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %1

; <label>:14
  ret void
}

; However, we can only merge ignore sign extensions when they are on all memory
; computations;
; CHECK: loadStoreBaseIndexOffsetSextNoSex
; CHECK-NOT: ldrh    [[REG:r[0-9]+]], [{{.*}}]
; CHECK-NOT: strh    [[REG]], [r1], #2
define void @loadStoreBaseIndexOffsetSextNoSex(ptr %a, ptr %b, ptr %c, i32 %n) {
  br label %1

; <label>:1
  %.09 = phi i32 [ %n, %0 ], [ %12, %1 ]
  %.08 = phi ptr [ %b, %0 ], [ %11, %1 ]
  %.0 = phi ptr [ %a, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i8, ptr %.0, i32 1
  %3 = load i8, ptr %.0, align 1
  %4 = sext i8 %3 to i32
  %5 = getelementptr inbounds i8, ptr %c, i32 %4
  %6 = load i8, ptr %5, align 1
  %7 = add i8 %3, 1
  %wrap.4 = sext i8 %7 to i32
  %8 = getelementptr inbounds i8, ptr %c, i32 %wrap.4
  %9 = load i8, ptr %8, align 1
  store i8 %6, ptr %.08, align 1
  %10 = getelementptr inbounds i8, ptr %.08, i32 1
  store i8 %9, ptr %10, align 1
  %11 = getelementptr inbounds i8, ptr %.08, i32 2
  %12 = add nsw i32 %.09, -1
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %1

; <label>:14
  ret void
}
