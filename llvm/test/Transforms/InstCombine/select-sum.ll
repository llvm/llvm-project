; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:64:64-p1:16:16-p2:32:32:32-p3:64:64:64"

; Test transform of:
;   ((P == A ? X : Def) + Base) + (P == B ? Y : 0)
; into:
;   Base + (P == A ? X : (Def + (P == B ? Y : 0))
define i32 @test_sum_selects(i32 %p, i32 %x, i32 %y, i32 %default, i32 %base) {
; CHECK-LABEL: @test_sum_selects
; CHECK-DAG: [[C0:%.*]] = icmp eq i32 %p, 288
; CHECK-DAG: [[C1:%.*]] = icmp eq i32 %p, 128
; CHECK-DAG: [[SELY:%.*]] = select i1 [[C1]], i32 %y, i32 0
; CHECK-DAG: [[SELY_DEF:%.*]] = add i32 %default, [[SELY]]
; CHECK-DAG: [[SELX:%.*]] = select i1 [[C0]], i32 %x, i32 [[SELY_DEF]]
; CHECK-DAG: [[SUM:%.*]] = add i32 %base, [[SELX]]
; CHECK-DAG: ret i32 [[SUM]]
  %c0 = icmp eq i32 %p, 288
  %sel0 = select i1 %c0, i32 %x, i32 %default
  %c1 = icmp eq i32 %p, 128
  %sel1 = select i1 %c1, i32 %y, i32 0
  %sum1 = add i32 %sel0, %base
  %sum2 = add i32 %sum1, %sel1
  ret i32 %sum2
}
