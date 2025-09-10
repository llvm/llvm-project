; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58831.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58831.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@i = dso_local local_unnamed_addr global ptr null, align 8
@b = dso_local local_unnamed_addr global ptr null, align 8
@a = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@p = dso_local local_unnamed_addr global i32 0, align 4
@q = dso_local local_unnamed_addr global i32 0, align 4
@r = dso_local global ptr null, align 8
@o = dso_local local_unnamed_addr global i16 0, align 4
@j = dso_local local_unnamed_addr global i16 0, align 2

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call fastcc void @fn2()
  store ptr @r, ptr @i, align 8, !tbaa !6
  %1 = load ptr, ptr @b, align 8, !tbaa !12
  tail call fastcc void @fn1(ptr noundef %1)
  ret i32 0
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define internal fastcc void @fn2() unnamed_addr #1 {
  store i16 0, ptr @o, align 4, !tbaa !14
  store i32 42, ptr @f, align 4, !tbaa !16
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define internal fastcc void @fn1(ptr noundef writeonly captures(none) %0) unnamed_addr #2 {
  %2 = load i32, ptr @p, align 4, !tbaa !16
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %8, label %4

4:                                                ; preds = %1, %4
  store i32 1, ptr %0, align 4, !tbaa !16
  %5 = load i32, ptr @p, align 4, !tbaa !16
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr @p, align 4, !tbaa !16
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %4, !llvm.loop !18

8:                                                ; preds = %4, %1
  store ptr @d, ptr @r, align 8, !tbaa !12
  store ptr @d, ptr @b, align 8, !tbaa !12
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noinline norecurse nosync nounwind memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p2 int", !8, i64 0}
!8 = !{!"any p2 pointer", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"p1 int", !9, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"short", !10, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !10, i64 0}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
