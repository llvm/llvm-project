; RUN: opt -S -indvars < %s | FileCheck %s

; Check that SCEV is able to recognize and use guards to prove
; conditions gaurding loop entries and backedges.  This isn't intended
; to be a comprehensive test of SCEV's simplification capabilities,
; tests directly testing e.g. if SCEV can elide a sext should go
; elsewhere.

target datalayout = "n8:16:32:64"

declare void @llvm.experimental.guard(i1, ...)

declare void @use(i64 %x)

define void @test_1(i1* %cond_buf, i32* %len_buf) {
; CHECK-LABEL: @test_1(
entry:
  %len = load i32, i32* %len_buf, !range !{i32 1, i32 2147483648}
  br label %loop

loop:
; CHECK: loop:
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
; CHECK:  %iv.inc.cmp = icmp ult i32 %iv.inc, %len
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %iv.inc.cmp) [ "deopt"() ]
; CHECK: leave:

  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1

  %iv.cmp = icmp slt i32 %iv, %len
  call void(i1, ...) @llvm.experimental.guard(i1 %iv.cmp) [ "deopt"() ]

  %iv.inc.cmp = icmp slt i32 %iv.inc, %len
  call void(i1, ...) @llvm.experimental.guard(i1 %iv.inc.cmp) [ "deopt"() ]

  %becond = load volatile i1, i1* %cond_buf
  br i1 %becond, label %loop, label %leave

leave:
  ret void
}

define void @test_2(i32 %n, i32* %len_buf) {
; CHECK-LABEL: @test_2(
; CHECK:  [[LEN_ZEXT:%[^ ]+]] = zext i32 %len to i64
; CHECK:  br label %loop

entry:
  %len = load i32, i32* %len_buf, !range !{i32 0, i32 2147483648}
  br label %loop

loop:
; CHECK: loop:
; CHECK:  %indvars.iv = phi i64 [ %indvars.iv.next, %loop ], [ 0, %entry ]
; CHECK:  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:  %iv.inc.cmp = icmp ult i64 %indvars.iv.next, [[LEN_ZEXT]]
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %iv.inc.cmp) [ "deopt"() ]
; CHECK: leave:

  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1

  %iv.sext = sext i32 %iv to i64
  call void @use(i64 %iv.sext)

  %iv.inc.cmp = icmp slt i32 %iv.inc, %len
  call void(i1, ...) @llvm.experimental.guard(i1 %iv.inc.cmp) [ "deopt"() ]

  %becond = icmp ne i32 %iv, %n
  br i1 %becond, label %loop, label %leave

leave:
  ret void
}

define void @test_3(i1* %cond_buf, i32* %len_buf) {
; CHECK-LABEL: @test_3(

entry:
  %len = load i32, i32* %len_buf
  %entry.cond = icmp sgt i32 %len, 0
  call void(i1, ...) @llvm.experimental.guard(i1 %entry.cond) [ "deopt"() ]
  br label %loop

loop:
; CHECK: loop:
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
; CHECK:  %iv.inc.cmp = icmp slt i32 %iv.inc, %len
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %iv.inc.cmp) [ "deopt"() ]
; CHECK: leave:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1

  %iv.cmp = icmp slt i32 %iv, %len
  call void(i1, ...) @llvm.experimental.guard(i1 %iv.cmp) [ "deopt"() ]

  %iv.inc.cmp = icmp slt i32 %iv.inc, %len
  call void(i1, ...) @llvm.experimental.guard(i1 %iv.inc.cmp) [ "deopt"() ]

  %becond = load volatile i1, i1* %cond_buf
  br i1 %becond, label %loop, label %leave

leave:
  ret void
}

define void @test_4(i1* %cond_buf, i32* %len_buf) {
; CHECK-LABEL: @test_4(

entry:
  %len = load i32, i32* %len_buf
  %entry.cond = icmp sgt i32 %len, 0
  call void(i1, ...) @llvm.experimental.guard(i1 %entry.cond) [ "deopt"() ]
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %be ]
  %iv.inc = add i32 %iv, 1

  %cond = load volatile i1, i1* %cond_buf
  br i1 %cond, label %left, label %be

left:
  ; Does not dominate the backedge, so cannot be used in the inductive proof
  %iv.inc.cmp = icmp slt i32 %iv.inc, %len
  call void(i1, ...) @llvm.experimental.guard(i1 %iv.inc.cmp) [ "deopt"() ]
  br label %be

be:
; CHECK: be:
; CHECK-NEXT:  %iv.cmp = icmp slt i32 %iv, %len
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %iv.cmp) [ "deopt"() ]
; CHECK: leave:

  %iv.cmp = icmp slt i32 %iv, %len
  call void(i1, ...) @llvm.experimental.guard(i1 %iv.cmp) [ "deopt"() ]

  %becond = load volatile i1, i1* %cond_buf
  br i1 %becond, label %loop, label %leave

leave:
  ret void
}
