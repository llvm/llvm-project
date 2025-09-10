; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030714-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030714-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.RenderBox = type { [6 x i32], i16, i16, ptr }
%struct.RenderStyle = type { %struct.NonInheritedFlags }
%struct.NonInheritedFlags = type { %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { i32 }

@false = dso_local local_unnamed_addr constant i8 0, align 1
@true = dso_local local_unnamed_addr constant i8 1, align 1
@g_this = dso_local local_unnamed_addr global %struct.RenderBox zeroinitializer, align 8
@g__style = dso_local local_unnamed_addr global %struct.RenderStyle zeroinitializer, align 4

; Function Attrs: nounwind uwtable
define dso_local void @RenderBox_setStyle(ptr noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 26
  %4 = load i16, ptr %3, align 2
  %5 = load i32, ptr %1, align 4
  %6 = and i32 %5, 262144
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %10, label %8

8:                                                ; preds = %2
  %9 = or i16 %4, 16
  br label %29

10:                                               ; preds = %2
  %11 = and i16 %4, -17
  store i16 %11, ptr %3, align 2
  %12 = load i32, ptr %1, align 4
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %14 = load ptr, ptr %13, align 8, !tbaa !6
  %15 = tail call i1 %14(ptr noundef nonnull %0) #4
  %16 = and i32 %12, 1572864
  %17 = icmp eq i32 %16, 0
  %18 = select i1 %15, i1 true, i1 %17
  br i1 %18, label %22, label %19

19:                                               ; preds = %10
  %20 = load i16, ptr %3, align 2
  %21 = or i16 %20, 8
  br label %29

22:                                               ; preds = %10
  %23 = load i32, ptr %1, align 4
  %24 = and i32 %23, 393216
  %25 = icmp eq i32 %24, 131072
  br i1 %25, label %26, label %31

26:                                               ; preds = %22
  %27 = load i16, ptr %3, align 2
  %28 = or i16 %27, 64
  br label %29

29:                                               ; preds = %8, %26, %19
  %30 = phi i16 [ %21, %19 ], [ %28, %26 ], [ %9, %8 ]
  store i16 %30, ptr %3, align 2
  br label %31

31:                                               ; preds = %29, %22
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @RenderObject_setStyle(ptr noundef readnone captures(none) %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @removeFromSpecialObjects(ptr noundef readnone captures(none) %0) local_unnamed_addr #1 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i1 @RenderBox_isTableCell(ptr readnone captures(none) %0) #1 {
  ret i1 false
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load i16, ptr getelementptr inbounds nuw (i8, ptr @g_this, i64 26), align 2
  %2 = and i16 %1, -89
  store ptr @RenderBox_isTableCell, ptr getelementptr inbounds nuw (i8, ptr @g_this, i64 32), align 8, !tbaa !6
  %3 = load i32, ptr @g__style, align 4
  %4 = and i32 %3, -1966081
  %5 = or disjoint i32 %4, 393216
  store i32 %5, ptr @g__style, align 4
  %6 = or disjoint i16 %2, 16
  store i16 %6, ptr getelementptr inbounds nuw (i8, ptr @g_this, i64 26), align 2
  tail call void @exit(i32 noundef 0) #5
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !12, i64 32}
!7 = !{!"RenderBox", !8, i64 0, !10, i64 24, !11, i64 26, !11, i64 26, !11, i64 26, !11, i64 26, !11, i64 26, !11, i64 26, !11, i64 26, !11, i64 26, !11, i64 27, !11, i64 27, !11, i64 27, !11, i64 27, !11, i64 27, !11, i64 27, !11, i64 27, !11, i64 27, !12, i64 32}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"short", !8, i64 0}
!11 = !{!"_Bool", !8, i64 0}
!12 = !{!"any pointer", !8, i64 0}
