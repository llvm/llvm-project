; RUN: opt -passes=loop-vectorize -S < %s | FileCheck %s

; NOTE: This is a focused reproducer for OpenMP declare-simd style VFABI mapping.

target triple = "x86_64-unknown-linux-gnu"

define i32 @test_vector_abi() local_unnamed_addr #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %src = getelementptr inbounds nuw double, ptr @c, i64 %iv
  %v = load double, ptr %src, align 8, !tbaa !4, !llvm.access.group !8
  %r = tail call double @acosh(double noundef %v) #2, !llvm.access.group !8
  %dst = getelementptr inbounds nuw double, ptr @x, i64 %iv
  store double %r, ptr %dst, align 8, !tbaa !4, !llvm.access.group !8
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, 1000
  br i1 %done, label %exit, label %loop, !llvm.loop !9

exit:
  ret i32 0
}

@x = dso_local local_unnamed_addr global [1000 x double] zeroinitializer, align 16
@c = dso_local local_unnamed_addr global [1000 x double] zeroinitializer, align 16

; CHECK-LABEL: @test_vector_abi(
; CHECK: vector.body:
; CHECK: call <2 x double> @_ZGVbN2v_acosh

declare double @acosh(double noundef) local_unnamed_addr #1

declare <2 x double> @_ZGVbN2v_acosh(<2 x double>)
declare <4 x double> @_ZGVcN4v_acosh(<4 x double>)
declare <4 x double> @_ZGVdN4v_acosh(<4 x double>)
declare <8 x double> @_ZGVeN8v_acosh(<8 x double>)

attributes #0 = { noinline nounwind strictfp }
attributes #1 = { nounwind "_ZGVbN2v_acosh" "_ZGVcN4v_acosh" "_ZGVdN4v_acosh" "_ZGVeN8v_acosh" "no-builtins" "vector-function-abi-variant"="_ZGVbN2v_acosh,_ZGVcN4v_acosh,_ZGVdN4v_acosh,_ZGVeN8v_acosh" }
attributes #2 = { nobuiltin nounwind strictfp "no-builtins" }

!4 = !{!5, !5, i64 0}
!5 = !{!"double", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.parallel_accesses", !8}
!11 = !{!"llvm.loop.vectorize.enable", i1 true}
