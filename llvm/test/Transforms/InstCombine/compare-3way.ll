; RUN: opt -S -passes=instcombine < %s | FileCheck %s

declare void @use(i32)

; These 18 exercise all combinations of signed comparison
; for each of the three values produced by your typical 
; 3way compare function (-1, 0, 1)

define void @test_low_sgt(i64 %a, i64 %b) {
; CHECK-LABEL: @test_low_sgt
; CHECK: [[TMP1:%.*]] = icmp slt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %normal, label %unreached
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sgt i32 %result, -1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_low_slt(i64 %a, i64 %b) {
; CHECK-LABEL: @test_low_slt
; CHECK: br i1 false, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp slt i32 %result, -1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_low_sge(i64 %a, i64 %b) {
; CHECK-LABEL: @test_low_sge
; CHECK: br i1 true, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sge i32 %result, -1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_low_sle(i64 %a, i64 %b) {
; CHECK-LABEL: @test_low_sle
; CHECK: [[TMP1:%.*]] = icmp slt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sle i32 %result, -1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_low_ne(i64 %a, i64 %b) {
; CHECK-LABEL: @test_low_ne
; CHECK: [[TMP1:%.*]] = icmp slt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %normal, label %unreached
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp ne i32 %result, -1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_low_eq(i64 %a, i64 %b) {
; CHECK-LABEL: @test_low_eq
; CHECK: [[TMP1:%.*]] = icmp slt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp eq i32 %result, -1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_mid_sgt(i64 %a, i64 %b) {
; CHECK-LABEL: @test_mid_sgt
; CHECK: [[TMP1:%.*]] = icmp sgt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sgt i32 %result, 0
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_mid_slt(i64 %a, i64 %b) {
; CHECK-LABEL: @test_mid_slt
; CHECK: [[TMP1:%.*]] = icmp slt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp slt i32 %result, 0
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_mid_sge(i64 %a, i64 %b) {
; CHECK-LABEL: @test_mid_sge
; CHECK: [[TMP1:%.*]] = icmp slt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %normal, label %unreached
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sge i32 %result, 0
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_mid_sle(i64 %a, i64 %b) {
; CHECK-LABEL: @test_mid_sle
; CHECK: [[TMP1:%.*]] = icmp sgt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %normal, label %unreached
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sle i32 %result, 0
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_mid_ne(i64 %a, i64 %b) {
; CHECK-LABEL: @test_mid_ne
; CHECK: [[TMP1:%.*]] = icmp eq i64 %a, %b
; CHECK: br i1 [[TMP1]], label %normal, label %unreached
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp ne i32 %result, 0
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_mid_eq(i64 %a, i64 %b) {
; CHECK-LABEL: @test_mid_eq
; CHECK: icmp eq i64 %a, %b
; CHECK: br i1 %eq, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp eq i32 %result, 0
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_high_sgt(i64 %a, i64 %b) {
; CHECK-LABEL: @test_high_sgt
; CHECK: br i1 false, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sgt i32 %result, 1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_high_slt(i64 %a, i64 %b) {
; CHECK-LABEL: @test_high_slt
; CHECK: [[TMP1:%.*]] = icmp sgt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %normal, label %unreached
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp slt i32 %result, 1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_high_sge(i64 %a, i64 %b) {
; CHECK-LABEL: @test_high_sge
; CHECK: [[TMP1:%.*]] = icmp sgt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sge i32 %result, 1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_high_sle(i64 %a, i64 %b) {
; CHECK-LABEL: @test_high_sle
; CHECK: br i1 true, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp sle i32 %result, 1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_high_ne(i64 %a, i64 %b) {
; CHECK-LABEL: @test_high_ne
; CHECK: [[TMP1:%.*]] = icmp sgt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %normal, label %unreached
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp ne i32 %result, 1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @test_high_eq(i64 %a, i64 %b) {
; CHECK-LABEL: @test_high_eq
; CHECK: [[TMP1:%.*]] = icmp sgt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -1, i32 1
  %result = select i1 %eq, i32 0, i32 %.
  %cmp = icmp eq i32 %result, 1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

; These five make sure we didn't accidentally hard code one of the
; produced values

define void @non_standard_low(i64 %a, i64 %b) {
; CHECK-LABEL: @non_standard_low
; CHECK: [[TMP1:%.*]] = icmp slt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -3, i32 -1
  %result = select i1 %eq, i32 -2, i32 %.
  %cmp = icmp eq i32 %result, -3
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @non_standard_mid(i64 %a, i64 %b) {
; CHECK-LABEL: @non_standard_mid
; CHECK: icmp eq i64 %a, %b
; CHECK: br i1 %eq, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -3, i32 -1
  %result = select i1 %eq, i32 -2, i32 %.
  %cmp = icmp eq i32 %result, -2
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @non_standard_high(i64 %a, i64 %b) {
; CHECK-LABEL: @non_standard_high
; CHECK: [[TMP1:%.*]] = icmp sgt i64 %a, %b
; CHECK: br i1 [[TMP1]], label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -3, i32 -1
  %result = select i1 %eq, i32 -2, i32 %.
  %cmp = icmp eq i32 %result, -1
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @non_standard_bound1(i64 %a, i64 %b) {
; CHECK-LABEL: @non_standard_bound1
; CHECK: br i1 false, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -3, i32 -1
  %result = select i1 %eq, i32 -2, i32 %.
  %cmp = icmp eq i32 %result, -20
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}

define void @non_standard_bound2(i64 %a, i64 %b) {
; CHECK-LABEL: @non_standard_bound2
; CHECK: br i1 false, label %unreached, label %normal
  %eq = icmp eq i64 %a, %b
  %slt = icmp slt i64 %a, %b
  %. = select i1 %slt, i32 -3, i32 -1
  %result = select i1 %eq, i32 -2, i32 %.
  %cmp = icmp eq i32 %result, 0
  br i1 %cmp, label %unreached, label %normal
normal:
  ret void
unreached:
  call void @use(i32 %result)
  ret void
}
