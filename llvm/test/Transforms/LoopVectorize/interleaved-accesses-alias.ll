; RUN: opt -S -loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true < %s | FileCheck %s

; When merging two stores with interleaved access vectorization, make sure we
; propagate the alias information from all scalar stores to form the most
; generic alias info.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

%struct.Vec4r = type { double, double, double, double }
%struct.Vec2r = type { double, double }

define void @foobar(%struct.Vec4r* nocapture readonly %p, i32 %i)
{
entry:
  %cp = alloca [20 x %struct.Vec2r], align 8
  %0 = bitcast [20 x %struct.Vec2r]* %cp to i8*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %arraydecay = getelementptr inbounds [20 x %struct.Vec2r], [20 x %struct.Vec2r]* %cp, i64 0, i64 0
  call void @g(%struct.Vec2r* nonnull %arraydecay) #4
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %x = getelementptr inbounds %struct.Vec4r, %struct.Vec4r* %p, i64 %indvars.iv, i32 0
  %1 = load double, double* %x, align 8, !tbaa !3
  %mul = fmul double %1, 2.000000e+00
  %x4 = getelementptr inbounds [20 x %struct.Vec2r], [20 x %struct.Vec2r]* %cp, i64 0, i64 %indvars.iv, i32 0

; The new store should alias any double rather than one of the fields of Vec2r.
; CHECK: store <4 x double> {{.*}} !tbaa ![[STORE_TBAA:[0-9]+]]
; CHECK-DAG: ![[DOUBLE_TBAA:[0-9]+]] = !{!"double", !{{[0-9+]}}, i64 0}
; CHECK-DAG: ![[STORE_TBAA]] = !{![[DOUBLE_TBAA]], ![[DOUBLE_TBAA]], i64 0}
  store double %mul, double* %x4, align 8, !tbaa !8
  %y = getelementptr inbounds %struct.Vec4r, %struct.Vec4r* %p, i64 %indvars.iv, i32 1
  %2 = load double, double* %y, align 8, !tbaa !10
  %mul7 = fmul double %2, 3.000000e+00
  %y10 = getelementptr inbounds [20 x %struct.Vec2r], [20 x %struct.Vec2r]* %cp, i64 0, i64 %indvars.iv, i32 1
  store double %mul7, double* %y10, align 8, !tbaa !11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 4
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare void @g(%struct.Vec2r*)

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 6.0.0 (trunk 319007) (llvm/trunk 319324)"}
!3 = !{!4, !5, i64 0}
!4 = !{!"Vec4r", !5, i64 0, !5, i64 8, !5, i64 16, !5, i64 24}
!5 = !{!"double", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !5, i64 0}
!9 = !{!"Vec2r", !5, i64 0, !5, i64 8}
!10 = !{!4, !5, i64 8}
!11 = !{!9, !5, i64 8}
