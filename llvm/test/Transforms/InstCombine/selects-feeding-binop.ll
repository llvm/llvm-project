; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:64:64-p1:16:16-p2:32:32:32-p3:64:64:64"

; Test transformation of:
;  (p == c0 ? X : 0) + (p == c1 ? Y : 0)
; into:
;  p == c0 ? X : (p == c1 : Y : 0)
define i32 @test_sum_select_zero(i32 %p, i32 %x, i32 %y) {
; CHECK-LABEL: @test_sum_select_zero
; CHECK-DAG: [[C0:%.*]] = icmp eq i32 %p, 288
; CHECK-DAG: [[C1:%.*]] = icmp eq i32 %p, 128
; CHECK-DAG: [[SELY:%.*]] = select i1 [[C1]], i32 %y, i32 0
; CHECK-DAG: [[SELX:%.*]] = select i1 [[C0]], i32 %x, i32 [[SELY]]
; CHECK-DAG: ret i32 [[SELX]]
  %c0 = icmp eq i32 %p, 288
  %s0 = select i1 %c0, i32 %x, i32 0
  %c1 = icmp eq i32 %p, 128
  %s1 = select i1 %c1, i32 %y, i32 0
  %sum = add i32 %s0, %s1
  ret i32 %sum
}

; Verify that resolving the nested select using the implied condition is not
; limited to addition.
define i32 @test_or_select_zero(i32 %p, i32 %x, i32 %y) {
; CHECK-LABEL: @test_or_select_zero(
; CHECK-NEXT:    [[C0:%.*]] = icmp eq i32 [[P:%.*]], 288
; CHECK-NEXT:    [[C1:%.*]] = icmp eq i32 [[P]], 128
; CHECK-NEXT:    [[S1:%.*]] = select i1 [[C1]], i32 [[Y:%.*]], i32 0
; CHECK-NEXT:    [[SUM:%.*]] = select i1 [[C0]], i32 [[X:%.*]], i32 [[S1]]
; CHECK-NEXT:    ret i32 [[SUM]]
;
  %c0 = icmp eq i32 %p, 288
  %s0 = select i1 %c0, i32 %x, i32 0
  %c1 = icmp eq i32 %p, 128
  %s1 = select i1 %c1, i32 %y, i32 0
  %sum = or i32 %s0, %s1
  ret i32 %sum
}
