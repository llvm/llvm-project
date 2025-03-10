; RUN: llc -o - %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-none-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

@llvm.compiler.used = appending global [1 x ptr] [ptr @_Z9BatchCastPKlPfi], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z9BatchCastPKlPfi(ptr noalias noundef readonly captures(none) %input, ptr noalias noundef writeonly captures(none) %output, i32 noundef %n) #0 {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  %min.iters.check = icmp ult i64 %wide.trip.count, 4
  br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %for.body.preheader
  %n.mod.vf = urem i64 %wide.trip.count, 4
  %n.vec = sub i64 %wide.trip.count, %n.mod.vf
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = add i64 %index, 0
  %1 = getelementptr inbounds nuw i64, ptr %input, i64 %0
  %2 = getelementptr inbounds nuw i64, ptr %1, i32 0
  %wide.load = load <4 x i64>, ptr %2, align 8, !tbaa !6
  ; CHECK-NOT:    scvtf {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  %3 = sitofp <4 x i64> %wide.load to <4 x float>
  %4 = getelementptr inbounds nuw float, ptr %output, i64 %0
  %5 = getelementptr inbounds nuw float, ptr %4, i32 0
  store <4 x float> %3, ptr %5, align 4, !tbaa !10
  %index.next = add nuw i64 %index, 4
  %6 = icmp eq i64 %index.next, %n.vec
  br i1 %6, label %middle.block, label %vector.body, !llvm.loop !12

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %wide.trip.count, %n.vec
  br i1 %cmp.n, label %for.cond.cleanup.loopexit, label %scalar.ph

scalar.ph:                                        ; preds = %for.body.preheader, %middle.block
  %bc.resume.val = phi i64 [ %n.vec, %middle.block ], [ 0, %for.body.preheader ]
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %middle.block, %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %scalar.ph, %for.body
  %indvars.iv = phi i64 [ %bc.resume.val, %scalar.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw i64, ptr %input, i64 %indvars.iv
  %7 = load i64, ptr %arrayidx, align 8, !tbaa !6
  %conv = sitofp i64 %7 to float
  %arrayidx2 = getelementptr inbounds nuw float, ptr %output, i64 %indvars.iv
  store float %conv, ptr %arrayidx2, align 4, !tbaa !10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !17
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 21.0.0git (git@github.com:llvm/llvm-project.git 46236f4c3dbe11e14fe7ac1f4b903637efedfecf)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !8, i64 0}
!12 = distinct !{!12, !13, !14, !15, !16}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.unroll.disable"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !13, !14, !15}
