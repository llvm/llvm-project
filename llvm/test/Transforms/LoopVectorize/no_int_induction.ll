; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

; int __attribute__((noinline)) sum_array(int *A, int n) {
;  return std::accumulate(A, A + n, 0);
; }

target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-n8:16:32:64-S128"

;CHECK-LABEL: @sum_array(
;CHECK: phi i64
;CHECK: phi <4 x i32>
;CHECK: load <4 x i32>
;CHECK: add <4 x i32>
;CHECK: ret i32
define i32 @sum_array(ptr %A, i32 %n) nounwind uwtable readonly noinline ssp {
  %1 = sext i32 %n to i64
  %2 = getelementptr inbounds i32, ptr %A, i64 %1
  %3 = icmp eq i32 %n, 0
  br i1 %3, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %.lr.ph.i

.lr.ph.i:                                         ; preds = %0, %.lr.ph.i
  %.03.i = phi ptr [ %6, %.lr.ph.i ], [ %A, %0 ]
  %.012.i = phi i32 [ %5, %.lr.ph.i ], [ 0, %0 ]
  %4 = load i32, ptr %.03.i, align 4
  %5 = add nsw i32 %4, %.012.i
  %6 = getelementptr inbounds i32, ptr %.03.i, i64 1
  %7 = icmp eq ptr %6, %2
  br i1 %7, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %.lr.ph.i

_ZSt10accumulateIPiiET0_T_S2_S1_.exit:            ; preds = %.lr.ph.i, %0
  %.01.lcssa.i = phi i32 [ 0, %0 ], [ %5, %.lr.ph.i ]
  ret i32 %.01.lcssa.i
}

; Same, but use a pointer with a different size.
;CHECK-LABEL: @sum_array_as1(
;CHECK: phi i16
;CHECK: phi <4 x i32>
;CHECK: load <4 x i32>
;CHECK: add <4 x i32>
;CHECK: ret i32
define i32 @sum_array_as1(ptr addrspace(1) %A, i32 %n) nounwind uwtable readonly noinline ssp {
  %1 = sext i32 %n to i64
  %2 = getelementptr inbounds i32, ptr addrspace(1) %A, i64 %1
  %3 = icmp eq i32 %n, 0
  br i1 %3, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %.lr.ph.i

.lr.ph.i:                                         ; preds = %0, %.lr.ph.i
  %.03.i = phi ptr addrspace(1) [ %6, %.lr.ph.i ], [ %A, %0 ]
  %.012.i = phi i32 [ %5, %.lr.ph.i ], [ 0, %0 ]
  %4 = load i32, ptr addrspace(1) %.03.i, align 4
  %5 = add nsw i32 %4, %.012.i
  %6 = getelementptr inbounds i32, ptr addrspace(1) %.03.i, i64 1
  %7 = icmp eq ptr addrspace(1) %6, %2
  br i1 %7, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %.lr.ph.i

_ZSt10accumulateIPiiET0_T_S2_S1_.exit:            ; preds = %.lr.ph.i, %0
  %.01.lcssa.i = phi i32 [ 0, %0 ], [ %5, %.lr.ph.i ]
  ret i32 %.01.lcssa.i
}
