; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-unknown-linux-android29"

@test_crash_format = dso_local local_unnamed_addr global i32 0, align 4
@global_out = external local_unnamed_addr global ptr, align 4

; Function Attrs: mustprogress nofree norecurse noreturn nosync nounwind optsize memory(readwrite, argmem: write, inaccessiblemem: none, target_mem: none)
define dso_local void @_Z10test_crashv() local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr @test_crash_format, align 4, !tbaa !5
  %1 = insertelement <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 poison, i32 poison, i32 poison, i32 undef>, i32 %0, i64 6
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %i.0 = phi i32 [ undef, %entry ], [ %inc, %for.cond ]
  %2 = insertelement <8 x i32> %1, i32 %i.0, i64 4
  %3 = insertelement <8 x i32> %2, i32 %i.0, i64 5
  %4 = load ptr, ptr @global_out, align 4, !tbaa !9
  store <8 x i32> %3, ptr %4, align 32, !tbaa !11
  %inc = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !12
}

attributes #0 = { mustprogress nofree norecurse noreturn nosync nounwind optsize memory(readwrite, argmem: write, inaccessiblemem: none, target_mem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+armv7-a,+d32,+dsp,+fp64,+neon,+read-tp-tpidruro,+thumb-mode,+vfp2,+vfp2sp,+vfp3,+vfp3d16,+vfp3d16sp,+vfp3sp,-aes,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fullfp16,-sha2,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 1, !"min_enum_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 23.0.0git (git@github.com:bababuck/llvm-project.git a6351d882ca3086c62e052fbc4f7f585e157213b)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !7, i64 0}
!11 = !{!7, !7, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
