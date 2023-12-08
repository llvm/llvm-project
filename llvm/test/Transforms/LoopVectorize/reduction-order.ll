; RUN: opt -passes='loop-vectorize' -force-vector-width=4 -force-vector-interleave=1 -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; Make sure the selects generated from reduction are always emitted
; in deterministic order.
; CHECK-LABEL: @foo(
; CHECK: vector.body:
; CHECK:      [[VEC_PHI_1:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[ADD_5:%.+]], %vector.body ]
; CHECK:      [[VEC_PHI_2:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[ADD_3:%.+]], %vector.body ]
; CHECK:      icmp ule <4 x i64>
; CHECK-NEXT: [[ADD_3]] = add <4 x i32> <i32 3, i32 3, i32 3, i32 3>, [[VEC_PHI_2]]
; CHECK-NEXT: [[ADD_5]] = add <4 x i32> [[VEC_PHI_1]], <i32 5, i32 5, i32 5, i32 5>
; CHECK:      select <4 x i1> {{.*}}, <4 x i32> [[ADD_5]], <4 x i32>
; CHECK-NEXT: select <4 x i1> {{.*}}, <4 x i32> [[ADD_3]], <4 x i32>
; CHECK: br i1 {{.*}}, label %middle.block, label %vector.body
;
define internal i64 @foo(ptr %t0) !prof !1 {
t16:
  br label %t20

t17:                                               ; preds = %t20
  %t18 = phi i32 [ %t24, %t20 ]
  %t19 = phi i32 [ %t28, %t20 ]
  br label %t31

t20:                                               ; preds = %t20, %t16
  %t21 = phi i64 [ 0, %t16 ], [ %t29, %t20 ]
  %t22 = phi i32 [ 0, %t16 ], [ %t28, %t20 ]
  %t23 = phi i32 [ 0, %t16 ], [ %t24, %t20 ]
  %t24 = add i32 3, %t23
  %t28 = add i32 %t22, 5
  %t29 = add nuw nsw i64 %t21, 1
  %t30 = icmp eq i64 %t29, 10
  br i1 %t30, label %t17, label %t20, !prof !2

t31:
  ret i64 undef
}

; Make sure we do not fail when checking for ordered reduction. This test just
; exercises the path and bails out without performing vectorization.
; CHECK-LABEL: quux
; CHECK-NOT: fadd <4 x 
define void @quux() {
bb:
  br label %header

latch:                                              ; preds = %header
  %tmp = phi double [ %tmp6, %header ]
  br i1 undef, label %header, label %bb2

bb2:                                              ; preds = %latch
  %tmp3 = phi double [ %tmp, %latch ]
  ret void

header:                                              ; preds = %latch, %bb
  %tmp5 = phi double [ 1.300000e+01, %bb ], [ %tmp, %latch ]
  %tmp6 = fadd double %tmp5, 1.000000e+00
  br label %latch
}

!1 = !{!"function_entry_count", i64 801}
!2 = !{!"branch_weights", i32 746, i32 1}
