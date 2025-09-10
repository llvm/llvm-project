; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020402-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020402-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.WorkEntrySType = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.ShrPcPteSType = type { %struct.ShrPcStatsSType }
%struct.ShrPcStatsSType = type { i32, i32, %struct.ShrPcCommonStatSType, %union.ShrPcStatUnion }
%struct.ShrPcCommonStatSType = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 }
%union.ShrPcStatUnion = type { %struct.ShrPcGemStatSType }
%struct.ShrPcGemStatSType = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, [40 x i64] }

@Local1 = dso_local local_unnamed_addr global ptr null, align 8
@Local2 = dso_local local_unnamed_addr global ptr null, align 8
@Local3 = dso_local local_unnamed_addr global ptr null, align 8
@RDbf1 = dso_local local_unnamed_addr global ptr null, align 8
@RDbf2 = dso_local local_unnamed_addr global ptr null, align 8
@RDbf3 = dso_local local_unnamed_addr global ptr null, align 8
@IntVc1 = dso_local local_unnamed_addr global ptr null, align 8
@IntVc2 = dso_local local_unnamed_addr global ptr null, align 8
@IntCode3 = dso_local local_unnamed_addr global ptr null, align 8
@IntCode4 = dso_local local_unnamed_addr global ptr null, align 8
@IntCode5 = dso_local local_unnamed_addr global ptr null, align 8
@IntCode6 = dso_local local_unnamed_addr global ptr null, align 8
@Lom1 = dso_local local_unnamed_addr global ptr null, align 8
@Lom2 = dso_local local_unnamed_addr global ptr null, align 8
@Lom3 = dso_local local_unnamed_addr global ptr null, align 8
@Lom4 = dso_local local_unnamed_addr global ptr null, align 8
@Lom5 = dso_local local_unnamed_addr global ptr null, align 8
@Lom6 = dso_local local_unnamed_addr global ptr null, align 8
@Lom7 = dso_local local_unnamed_addr global ptr null, align 8
@Lom8 = dso_local local_unnamed_addr global ptr null, align 8
@Lom9 = dso_local local_unnamed_addr global ptr null, align 8
@Lom10 = dso_local local_unnamed_addr global ptr null, align 8
@RDbf11 = dso_local local_unnamed_addr global ptr null, align 8
@RDbf12 = dso_local local_unnamed_addr global ptr null, align 8
@Workspace = dso_local local_unnamed_addr global %struct.WorkEntrySType zeroinitializer, align 8
@MyPte = dso_local global %struct.ShrPcPteSType zeroinitializer, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @InitCache(i32 noundef %0) local_unnamed_addr #0 {
  store i32 %0, ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 4), align 4, !tbaa !6
  store <2 x i64> <i64 0, i64 5>, ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 8), align 8, !tbaa !13
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 24), ptr @Local1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 32), ptr @Local2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 40), ptr @Local3, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 48), ptr @RDbf1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 56), ptr @RDbf2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 64), ptr @RDbf3, align 8, !tbaa !14
  store i64 1, ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 64), align 8, !tbaa !13
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 304), ptr @IntVc1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 312), ptr @IntVc2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 320), ptr @IntCode3, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 328), ptr @IntCode4, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 336), ptr @IntCode5, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 344), ptr @IntCode6, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 352), ptr @Workspace, align 8, !tbaa !17
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 360), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 8), align 8, !tbaa !19
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 368), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 16), align 8, !tbaa !20
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 376), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 24), align 8, !tbaa !21
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 384), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 32), align 8, !tbaa !22
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 392), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 40), align 8, !tbaa !23
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 400), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 48), align 8, !tbaa !24
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 408), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 56), align 8, !tbaa !25
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 416), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 64), align 8, !tbaa !26
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 424), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 72), align 8, !tbaa !27
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 432), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 80), align 8, !tbaa !28
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 208), ptr @Lom1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 216), ptr @Lom2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 224), ptr @Lom3, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 232), ptr @Lom4, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 240), ptr @Lom5, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 248), ptr @Lom6, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 256), ptr @Lom7, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 264), ptr @Lom8, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 272), ptr @Lom9, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 280), ptr @Lom10, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 288), ptr @RDbf11, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 296), ptr @RDbf12, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  store i32 5, ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 4), align 4, !tbaa !6
  store <2 x i64> <i64 0, i64 5>, ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 8), align 8, !tbaa !13
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 24), ptr @Local1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 32), ptr @Local2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 40), ptr @Local3, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 48), ptr @RDbf1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 56), ptr @RDbf2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 64), ptr @RDbf3, align 8, !tbaa !14
  store i64 1, ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 64), align 8, !tbaa !13
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 304), ptr @IntVc1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 312), ptr @IntVc2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 320), ptr @IntCode3, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 328), ptr @IntCode4, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 336), ptr @IntCode5, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 344), ptr @IntCode6, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 352), ptr @Workspace, align 8, !tbaa !17
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 360), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 8), align 8, !tbaa !19
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 368), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 16), align 8, !tbaa !20
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 376), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 24), align 8, !tbaa !21
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 384), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 32), align 8, !tbaa !22
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 392), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 40), align 8, !tbaa !23
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 400), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 48), align 8, !tbaa !24
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 408), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 56), align 8, !tbaa !25
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 416), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 64), align 8, !tbaa !26
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 424), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 72), align 8, !tbaa !27
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 432), ptr getelementptr inbounds nuw (i8, ptr @Workspace, i64 80), align 8, !tbaa !28
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 208), ptr @Lom1, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 216), ptr @Lom2, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 224), ptr @Lom3, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 232), ptr @Lom4, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 240), ptr @Lom5, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 248), ptr @Lom6, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 256), ptr @Lom7, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 264), ptr @Lom8, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 272), ptr @Lom9, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 280), ptr @Lom10, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 288), ptr @RDbf11, align 8, !tbaa !14
  store ptr getelementptr inbounds nuw (i8, ptr @MyPte, i64 296), ptr @RDbf12, align 8, !tbaa !14
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 4}
!7 = !{!"", !8, i64 0, !8, i64 4, !11, i64 8, !9, i64 208}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"", !12, i64 0, !12, i64 8, !12, i64 16, !12, i64 24, !12, i64 32, !12, i64 40, !12, i64 48, !12, i64 56, !12, i64 64, !12, i64 72, !12, i64 80, !12, i64 88, !12, i64 96, !12, i64 104, !12, i64 112, !12, i64 120, !12, i64 128, !12, i64 136, !12, i64 144, !12, i64 152, !12, i64 160, !12, i64 168, !12, i64 176, !12, i64 184, !12, i64 192}
!12 = !{!"long", !9, i64 0}
!13 = !{!12, !12, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 long", !16, i64 0}
!16 = !{!"any pointer", !9, i64 0}
!17 = !{!18, !15, i64 0}
!18 = !{!"", !15, i64 0, !15, i64 8, !15, i64 16, !15, i64 24, !15, i64 32, !15, i64 40, !15, i64 48, !15, i64 56, !15, i64 64, !15, i64 72, !15, i64 80}
!19 = !{!18, !15, i64 8}
!20 = !{!18, !15, i64 16}
!21 = !{!18, !15, i64 24}
!22 = !{!18, !15, i64 32}
!23 = !{!18, !15, i64 40}
!24 = !{!18, !15, i64 48}
!25 = !{!18, !15, i64 56}
!26 = !{!18, !15, i64 64}
!27 = !{!18, !15, i64 72}
!28 = !{!18, !15, i64 80}
