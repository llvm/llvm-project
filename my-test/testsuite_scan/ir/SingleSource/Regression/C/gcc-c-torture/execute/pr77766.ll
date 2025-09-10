; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr77766.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr77766.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@d = dso_local local_unnamed_addr global i16 5, align 4
@f = dso_local local_unnamed_addr global i32 4, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global [1 x i8] zeroinitializer, align 1
@b = dso_local local_unnamed_addr global i16 0, align 4
@j = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i8 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i16 0, align 2

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: write) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @f, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %21, label %3

3:                                                ; preds = %0
  %4 = load i16, ptr @d, align 4, !tbaa !10
  %5 = icmp slt i16 %4, 1
  %6 = load i32, ptr @j, align 4
  %7 = icmp eq i32 %6, 0
  %8 = load i8, ptr @a, align 4
  %9 = zext i8 %8 to i32
  %10 = icmp eq i8 %8, 0
  br i1 %5, label %11, label %19

11:                                               ; preds = %3
  %12 = load i16, ptr @b, align 4
  %13 = sext i16 %12 to i64
  %14 = getelementptr inbounds i8, ptr @c, i64 %13
  %15 = load i8, ptr %14, align 1, !tbaa !12
  %16 = icmp eq i8 %15, 0
  br i1 %16, label %17, label %19, !llvm.loop !13

17:                                               ; preds = %11
  store i32 0, ptr @g, align 4, !tbaa !6
  br label %18

18:                                               ; preds = %18, %17
  br label %18

19:                                               ; preds = %3, %11
  store i32 %9, ptr @f, align 4, !tbaa !6
  %20 = select i1 %7, i32 33, i32 0
  tail call void @llvm.assume(i1 %10)
  store i32 %20, ptr @g, align 4, !tbaa !6
  br label %21

21:                                               ; preds = %19, %0
  store i32 0, ptr @e, align 4, !tbaa !6
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"short", !8, i64 0}
!12 = !{!8, !8, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
