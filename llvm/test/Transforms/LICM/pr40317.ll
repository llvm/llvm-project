; RUN: opt -S -mcpu=z13 -passes=licm -licm-control-flow-hoisting -verify-memoryssa < %s | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

%0 = type { %1, %1, i16, %2 }
%1 = type <{ i16, i8, i32, i32, i32, i64, i64 }>
%2 = type { i8, i16, i16, [2 x i8] }

@0 = internal global %0 { %1 <{ i16 22437, i8 117, i32 2017322857, i32 900074563, i32 -1390364, i64 0, i64 0 }>, %1 <{ i16 0, i8 7, i32 -387299562, i32 925371866, i32 -1, i64 4826244575317081679, i64 1 }>, i16 8, %2 { i8 0, i16 0, i16 3, [2 x i8] undef } }, align 2
@g_18 = external dso_local global i64, align 8

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #1

; CHECK-LABEL: @func_94
; CHECK: bb:
; CHECK: tail call void @llvm.memset.p0.i64
; CHECK: load i32
; CHECK: bb6.licm:
; Function Attrs: noreturn nounwind
define dso_local void @func_94(i16 %arg, ptr nocapture %arg1) local_unnamed_addr #3 {
bb:
  tail call void @llvm.memset.p0.i64(ptr align 8 undef, i8 0, i64 80, i1 false)
  br label %bb3

bb3:                                              ; preds = %bb13, %bb
  %tmp5 = icmp eq i16 %arg, 0
  br i1 %tmp5, label %bb6, label %bb13

bb6:                                              ; preds = %bb3
  %tmp7 = load i32, ptr getelementptr inbounds (%0, ptr @0, i64 0, i32 1, i32 2), align 1, !tbaa !11
  %tmp8 = zext i32 %tmp7 to i64
  %sext = shl i64 %tmp8, 56
  %tmp10 = ashr exact i64 %sext, 56
  store i64 %tmp10, ptr %arg1, align 8, !tbaa !12
  br label %bb13

bb13:                                             ; preds = %bb3, %bb6
  br label %bb3
}

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { norecurse nounwind readnone "use-soft-float"="false" }
attributes #3 = { noreturn nounwind "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 8.0.0 (http://llvm.org/git/clang.git e593a791f2cf19db84237b0b9d632e9966a00a39) (http://llvm.org/git/llvm.git fe0523d1bd7def3ef62cfb3dd37a8b1941aafa81)"}
!1 = !{!2, !8, i64 46}
!2 = !{!"S5", !3, i64 0, !3, i64 31, !4, i64 62, !9, i64 64}
!3 = !{!"S2", !4, i64 0, !5, i64 2, !7, i64 3, !7, i64 7, !7, i64 11, !8, i64 15, !8, i64 23}
!4 = !{!"short", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"int", !5, i64 0}
!8 = !{!"long", !5, i64 0}
!9 = !{!"S3", !7, i64 0, !4, i64 2, !4, i64 4}
!10 = !{!2, !7, i64 42}
!11 = !{!2, !7, i64 34}
!12 = !{!8, !8, i64 0}
