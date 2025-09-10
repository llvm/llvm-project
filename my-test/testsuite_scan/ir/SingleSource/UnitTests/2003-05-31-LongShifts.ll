; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-31-LongShifts.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-31-LongShifts.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon = type { i64, i32 }

@.str = private unnamed_addr constant [15 x i8] c"0x%llx op %d:\0A\00", align 1
@.str.1 = private unnamed_addr constant [45 x i8] c"  ashr: 0x%llx\0A  lshr: 0x%llx\0A  shl: 0x%llx\0A\00", align 1
@Vals = dso_local global [8 x { i64, i32, [4 x i8] }] [{ i64, i32, [4 x i8] } { i64 123, i32 4, [4 x i8] zeroinitializer }, { i64, i32, [4 x i8] } { i64 123, i32 34, [4 x i8] zeroinitializer }, { i64, i32, [4 x i8] } { i64 -4, i32 4, [4 x i8] zeroinitializer }, { i64, i32, [4 x i8] } { i64 -5, i32 34, [4 x i8] zeroinitializer }, { i64, i32, [4 x i8] } { i64 -6000000000, i32 4, [4 x i8] zeroinitializer }, { i64, i32, [4 x i8] } { i64 -6000000000, i32 34, [4 x i8] zeroinitializer }, { i64, i32, [4 x i8] } { i64 6000000000, i32 4, [4 x i8] zeroinitializer }, { i64, i32, [4 x i8] } { i64 6000000000, i32 34, [4 x i8] zeroinitializer }], align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @Test(i64 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %0, i32 noundef %1)
  %4 = zext i32 %1 to i64
  %5 = ashr i64 %0, %4
  %6 = lshr i64 %0, %4
  %7 = shl i64 %0, %4
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i64 noundef %5, i64 noundef %6, i64 noundef %7)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %0, -1
  %4 = icmp ult i32 %3, 8
  br i1 %4, label %5, label %22

5:                                                ; preds = %2
  %6 = zext nneg i32 %3 to i64
  br label %7

7:                                                ; preds = %5, %7
  %8 = phi i64 [ %6, %5 ], [ %19, %7 ]
  %9 = getelementptr inbounds nuw %struct.anon, ptr @Vals, i64 %8
  %10 = load volatile i64, ptr %9, align 8, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %12 = load volatile i32, ptr %11, align 8, !tbaa !12
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %10, i32 noundef %12)
  %14 = zext i32 %12 to i64
  %15 = ashr i64 %10, %14
  %16 = lshr i64 %10, %14
  %17 = shl i64 %10, %14
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i64 noundef %15, i64 noundef %16, i64 noundef %17)
  %19 = add nuw nsw i64 %8, 1
  %20 = and i64 %19, 4294967295
  %21 = icmp eq i64 %20, 8
  br i1 %21, label %22, label %7, !llvm.loop !13

22:                                               ; preds = %7, %2
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"", !8, i64 0, !11, i64 8}
!8 = !{!"long long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"int", !9, i64 0}
!12 = !{!7, !11, i64 8}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
