; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/2003-05-21-BitfieldHandling.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/2003-05-21-BitfieldHandling.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.test1 = type { i8, [7 x i8] }
%struct.test2 = type { i8, [7 x i8] }
%struct.test3 = type { i32, [4 x i8] }
%struct.test4 = type { i64 }
%struct.test5 = type { i32, [4 x i8] }
%struct.test6 = type { i64 }
%struct.test = type { i8, i8, [2 x i8], i8, i8, [2 x i8] }
%struct.test_empty = type {}

@Esize = dso_local local_unnamed_addr global i32 0, align 4
@N = dso_local local_unnamed_addr global { i16, i8, i8, [4 x i8], i8, i8, i8, i8, i8, i8, i8, i8 } { i16 2, i8 56, i8 0, [4 x i8] zeroinitializer, i8 1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0 }, align 8
@Nsize = dso_local local_unnamed_addr global i32 16, align 4
@F1size = dso_local local_unnamed_addr global i32 8, align 4
@F2size = dso_local local_unnamed_addr global i32 8, align 4
@F3size = dso_local local_unnamed_addr global i32 8, align 4
@F4size = dso_local local_unnamed_addr global i32 8, align 4
@F5size = dso_local local_unnamed_addr global i32 8, align 4
@F6size = dso_local local_unnamed_addr global i32 8, align 4
@Msize = dso_local local_unnamed_addr global i32 8, align 4
@.str = private unnamed_addr constant [16 x i8] c"N: %d %d %d %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"F1: %d\0A\00", align 1
@F1 = dso_local local_unnamed_addr global %struct.test1 zeroinitializer, align 8
@.str.2 = private unnamed_addr constant [8 x i8] c"F2: %d\0A\00", align 1
@F2 = dso_local local_unnamed_addr global %struct.test2 zeroinitializer, align 8
@.str.3 = private unnamed_addr constant [8 x i8] c"F3: %d\0A\00", align 1
@F3 = dso_local local_unnamed_addr global %struct.test3 zeroinitializer, align 8
@.str.4 = private unnamed_addr constant [11 x i8] c"F4: %d %d\0A\00", align 1
@F4 = dso_local local_unnamed_addr global %struct.test4 zeroinitializer, align 8
@.str.5 = private unnamed_addr constant [11 x i8] c"F5: %d %d\0A\00", align 1
@F5 = dso_local local_unnamed_addr global %struct.test5 zeroinitializer, align 8
@.str.6 = private unnamed_addr constant [11 x i8] c"F6: %d %d\0A\00", align 1
@F6 = dso_local local_unnamed_addr global %struct.test6 zeroinitializer, align 8
@.str.7 = private unnamed_addr constant [19 x i8] c"M: %d %d %d %d %d\0A\00", align 1
@M = dso_local local_unnamed_addr global %struct.test zeroinitializer, align 8
@e = dso_local local_unnamed_addr global %struct.test_empty zeroinitializer, align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i16, ptr @N, align 8, !tbaa !6
  %2 = zext i16 %1 to i32
  %3 = load i16, ptr getelementptr inbounds nuw (i8, ptr @N, i64 2), align 2
  %4 = shl i16 %3, 5
  %5 = ashr i16 %4, 8
  %6 = sext i16 %5 to i32
  %7 = load i64, ptr getelementptr inbounds nuw (i8, ptr @N, i64 8), align 8
  %8 = shl i64 %7, 33
  %9 = ashr exact i64 %8, 33
  %10 = trunc nsw i64 %9 to i32
  %11 = shl i64 %7, 2
  %12 = ashr i64 %11, 33
  %13 = trunc nsw i64 %12 to i32
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %2, i32 noundef %6, i32 noundef %10, i32 noundef %13)
  %15 = load i8, ptr @F1, align 8
  %16 = and i8 %15, 1
  %17 = zext nneg i8 %16 to i32
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %17)
  %19 = load i8, ptr @F2, align 8
  %20 = shl i8 %19, 4
  %21 = ashr exact i8 %20, 4
  %22 = sext i8 %21 to i32
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %22)
  %24 = load i32, ptr @F3, align 8
  %25 = and i32 %24, 1
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %25)
  %27 = load i64, ptr @F4, align 8
  %28 = trunc i64 %27 to i32
  %29 = and i32 %28, 1
  %30 = shl i64 %27, 18
  %31 = ashr i64 %30, 50
  %32 = trunc nsw i64 %31 to i32
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %29, i32 noundef %32)
  %34 = load i32, ptr @F5, align 8
  %35 = and i32 %34, 1
  %36 = lshr i32 %34, 18
  %37 = and i32 %36, 1
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %35, i32 noundef %37)
  %39 = load i64, ptr @F6, align 8
  %40 = trunc i64 %39 to i32
  %41 = and i32 %40, 1
  %42 = ashr i64 %39, 43
  %43 = trunc nsw i64 %42 to i32
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef %41, i32 noundef %43)
  %45 = load i8, ptr @M, align 8, !tbaa !13
  %46 = zext i8 %45 to i32
  %47 = load i8, ptr getelementptr inbounds nuw (i8, ptr @M, i64 1), align 1
  %48 = and i8 %47, 7
  %49 = zext nneg i8 %48 to i32
  %50 = lshr i8 %47, 3
  %51 = and i8 %50, 7
  %52 = zext nneg i8 %51 to i32
  %53 = load i8, ptr getelementptr inbounds nuw (i8, ptr @M, i64 4), align 4, !tbaa !15
  %54 = zext i8 %53 to i32
  %55 = load i8, ptr getelementptr inbounds nuw (i8, ptr @M, i64 5), align 1
  %56 = shl i8 %55, 4
  %57 = ashr exact i8 %56, 4
  %58 = sext i8 %57 to i32
  %59 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %46, i32 noundef %49, i32 noundef %52, i32 noundef %54, i32 noundef %58)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

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
!7 = !{!"rtx_def", !8, i64 0, !11, i64 2, !12, i64 8, !12, i64 11}
!8 = !{!"short", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"int", !9, i64 0}
!12 = !{!"long long", !9, i64 0}
!13 = !{!14, !9, i64 0}
!14 = !{!"test", !9, i64 0, !9, i64 1, !9, i64 1, !9, i64 4, !12, i64 5}
!15 = !{!14, !9, i64 4}
