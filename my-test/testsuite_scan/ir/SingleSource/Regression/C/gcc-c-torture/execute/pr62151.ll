; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr62151.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr62151.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i16 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @fn1() local_unnamed_addr #0 {
  %1 = alloca [2 x i32], align 4
  store i16 0, ptr @b, align 4, !tbaa !6
  %2 = load i32, ptr @h, align 4, !tbaa !10
  %3 = icmp eq i32 %2, 0
  %4 = load i32, ptr @f, align 4, !tbaa !10
  br i1 %3, label %5, label %16

5:                                                ; preds = %0
  %6 = load i32, ptr @c, align 4
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %15, !llvm.loop !12

8:                                                ; preds = %5
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #2
  %9 = sext i32 %4 to i64
  %10 = getelementptr inbounds i32, ptr %1, i64 %9
  store i32 0, ptr %10, align 4, !tbaa !10
  %11 = icmp eq i32 %4, 0
  br i1 %11, label %13, label %12

12:                                               ; preds = %8
  store i32 0, ptr @f, align 4, !tbaa !10
  br label %13

13:                                               ; preds = %12, %8
  %14 = load i32, ptr %1, align 4, !tbaa !10
  store i32 -1, ptr @i, align 4
  store i32 -1, ptr @g, align 4
  store i32 %14, ptr @e, align 4
  store i32 0, ptr @a, align 4, !tbaa !10
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #2
  ret i32 0

15:                                               ; preds = %5, %15
  br label %15

16:                                               ; preds = %0
  store i32 0, ptr @d, align 4, !tbaa !10
  br label %17

17:                                               ; preds = %17, %16
  br label %17
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [2 x i32], align 4
  store i16 0, ptr @b, align 4, !tbaa !6
  %2 = load i32, ptr @h, align 4, !tbaa !10
  %3 = icmp eq i32 %2, 0
  %4 = load i32, ptr @f, align 4, !tbaa !10
  br i1 %3, label %5, label %14

5:                                                ; preds = %0
  %6 = load i32, ptr @c, align 4
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %13, !llvm.loop !12

8:                                                ; preds = %5
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #2
  %9 = sext i32 %4 to i64
  %10 = getelementptr inbounds i32, ptr %1, i64 %9
  store i32 0, ptr %10, align 4, !tbaa !10
  %11 = icmp eq i32 %4, 0
  br i1 %11, label %16, label %12

12:                                               ; preds = %8
  store i32 0, ptr @f, align 4, !tbaa !10
  br label %16

13:                                               ; preds = %5, %13
  br label %13

14:                                               ; preds = %0
  store i32 0, ptr @d, align 4, !tbaa !10
  br label %15

15:                                               ; preds = %15, %14
  br label %15

16:                                               ; preds = %12, %8
  %17 = load i32, ptr %1, align 4, !tbaa !10
  store i32 -1, ptr @i, align 4
  store i32 -1, ptr @g, align 4
  store i32 %17, ptr @e, align 4
  store i32 0, ptr @a, align 4, !tbaa !10
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #2
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
