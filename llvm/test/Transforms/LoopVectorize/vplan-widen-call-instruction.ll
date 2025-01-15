; RUN: opt -passes=loop-vectorize -force-vector-width=4 -enable-vplan-native-path -S %s | FileCheck %s

; Test that VPlan native path is able to widen call intructions like
; llvm.sqrt.* intrincis calls.

declare double @llvm.sqrt.f64(double %0)
define void @widen_call_instruction(ptr noalias nocapture readonly %a.in, ptr noalias nocapture readonly %b.in, ptr noalias nocapture %c.out) {
; CHECK-LABEL: @widen_call_instruction(

; CHECK: vector.body:
; CHECK-NEXT: %[[FOR1_INDEX:.*]] = phi i64 [ 0, %[[LABEL_PR:.*]] ], [ %{{.*}}, %[[LABEL_FOR1_LATCH:.*]] ]
; CHECK: %[[VEC_INDEX:.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %[[LABEL_PR]] ], [ %{{.*}}, %[[LABEL_FOR1_LATCH]] ]
; CHECK-NEXT: %[[A_PTR:.*]] = getelementptr inbounds double, ptr %a.in, <4 x i64> %[[VEC_INDEX]]
; CHECK-NEXT: %[[MASKED_GATHER1:.*]] = call <4 x double> @llvm.masked.gather.v4f64.v4p0(<4 x ptr> %[[A_PTR]], i32 8, <4 x i1> splat (i1 true), <4 x double> poison)
; CHECK-NEXT: %[[B_PTR:.*]] = getelementptr inbounds double, ptr %b.in, <4 x i64> %[[VEC_INDEX]]
; CHECK-NEXT: %[[MASKED_GATHER2:.*]] = call <4 x double> @llvm.masked.gather.v4f64.v4p0(<4 x ptr> %[[B_PTR]], i32 8, <4 x i1> splat (i1 true), <4 x double> poison)
; CHECK-NEXT: %[[B_SQRT:.*]] = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %[[MASKED_GATHER2]])
; CHECK-NEXT: br label %[[FOR2_HEADER:.*]]

; CHECK: [[FOR2_HEADER]]:
; CHECK-NEXT: %[[FOR2_INDEX:.*]] = phi <4 x i32> [ zeroinitializer, %vector.body ], [ %[[FOR2_INDEX_NEXT:.*]], %[[FOR2_HEADER]] ]
; CHECK-NEXT: %[[REDUCTION:.*]] = phi <4 x double> [ %[[MASKED_GATHER1]], %vector.body ], [ %[[REDUCTION_NEXT:.*]], %[[FOR2_HEADER]] ]
; CHECK-NEXT: %[[REDUCTION_NEXT]] = fadd <4 x double> %[[B_SQRT]], %[[REDUCTION]]
; CHECK-NEXT: %[[FOR2_INDEX_NEXT]] = add nuw nsw <4 x i32> %[[FOR2_INDEX]], splat (i32 1)
; CHECK-NEXT: %[[VEC_PTR:.*]] = icmp eq <4 x i32> %[[FOR2_INDEX_NEXT]], splat (i32 10000)
; CHECK-NEXT: %[[EXIT_COND:.*]] = extractelement <4 x i1> %[[VEC_PTR]], i32 0
; CHECK-NEXT: br i1 %[[EXIT_COND]], label %[[FOR1_LATCH:.*]], label %{{.*}}

; CHECK: [[FOR1_LATCH]]:
; CHECK-NEXT: %[[REDUCTION:.*]] = phi <4 x double> [ %[[REDUCTION_NEXT]], %[[FOR2_HEADER]] ]
; CHECK-NEXT: %[[C_PTR:.*]] = getelementptr inbounds double, ptr %c.out, <4 x i64> %[[VEC_INDEX]]
; CHECK-NEXT: call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> %[[REDUCTION]], <4 x ptr> %[[C_PTR]], i32 8, <4 x i1> splat (i1 true))
; CHECK-NEXT: %[[FOR1_INDEX_NEXT:.*]] = add nuw i64 %[[FOR1_INDEX]], 4
; CHECK-NEXT: %{{.*}} = add <4 x i64> %[[VEC_INDEX]], splat (i64 4)
; CHECK-NEXT: %[[EXIT_COND:.*]] = icmp eq i64 %[[FOR1_INDEX_NEXT]], 1000
; CHECK-NEXT: br i1 %[[EXIT_COND]], label %{{.*}}, label %vector.body

entry:
  br label %for1.header

for1.header:
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar11, %for1.latch ]
  %a.ptr = getelementptr inbounds double, ptr %a.in, i64 %indvar1
  %a = load double, ptr %a.ptr, align 8
  %b.ptr = getelementptr inbounds double, ptr %b.in, i64 %indvar1
  %b = load double, ptr %b.ptr, align 8
  %b.sqrt = call double @llvm.sqrt.f64(double %b)
  br label %for2.header

for2.header:
  %indvar2 = phi i32 [ 0, %for1.header ], [ %indvar21, %for2.header ]
  %a.reduction = phi double [ %a, %for1.header ], [ %a.reduction1, %for2.header ]
  %a.reduction1 = fadd double %b.sqrt, %a.reduction
  %indvar21 = add nuw nsw i32 %indvar2, 1
  %for2.cond = icmp eq i32 %indvar21, 10000
  br i1 %for2.cond, label %for1.latch, label %for2.header

for1.latch:
  %c.ptr = getelementptr inbounds double, ptr %c.out, i64 %indvar1
  store double %a.reduction1, ptr %c.ptr, align 8
  %indvar11 = add nuw nsw i64 %indvar1, 1
  %for1.cond = icmp eq i64 %indvar11, 1000
  br i1 %for1.cond, label %exit, label %for1.header, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
