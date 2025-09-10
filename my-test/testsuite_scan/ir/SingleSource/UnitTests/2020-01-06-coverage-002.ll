; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-002.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-002.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@x = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i16 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i8 0, align 4
@y = dso_local local_unnamed_addr global i64 0, align 8
@z = dso_local local_unnamed_addr global i64 0, align 8
@.str = private unnamed_addr constant [8 x i8] c"a = %i\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"b = %i\0A\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"c = %i\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"d = %i\0A\00", align 1
@.str.4 = private unnamed_addr constant [8 x i8] c"x = %i\0A\00", align 1
@.str.5 = private unnamed_addr constant [8 x i8] c"e = %i\0A\00", align 1
@.str.6 = private unnamed_addr constant [9 x i8] c"z = %li\0A\00", align 1
@.str.7 = private unnamed_addr constant [9 x i8] c"y = %li\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i64 @f() local_unnamed_addr #0 {
  store i32 0, ptr @x, align 4, !tbaa !6
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %6

3:                                                ; preds = %0
  %4 = load i16, ptr @d, align 4
  %5 = icmp eq i16 %4, 0
  br i1 %5, label %7, label %8

6:                                                ; preds = %0, %6
  br label %6

7:                                                ; preds = %3, %7
  br label %7

8:                                                ; preds = %3
  %9 = load i32, ptr @e, align 4, !tbaa !6
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %12

11:                                               ; preds = %8
  store i64 0, ptr @y, align 8, !tbaa !10
  br label %12

12:                                               ; preds = %8, %11
  ret i64 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i16 10, ptr @d, align 4, !tbaa !12
  store i8 0, ptr @b, align 4, !tbaa !14
  store i32 0, ptr @e, align 4, !tbaa !6
  store i32 -2, ptr @c, align 4, !tbaa !6
  store i32 0, ptr @a, align 4, !tbaa !6
  store i32 0, ptr @x, align 4, !tbaa !6
  store i64 0, ptr @y, align 8, !tbaa !10
  store i64 0, ptr @z, align 8, !tbaa !10
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0)
  %2 = load i8, ptr @b, align 4, !tbaa !14
  %3 = sext i8 %2 to i32
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %3)
  %5 = load i32, ptr @c, align 4, !tbaa !6
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %5)
  %7 = load i16, ptr @d, align 4, !tbaa !12
  %8 = sext i16 %7 to i32
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %8)
  %10 = load i32, ptr @x, align 4, !tbaa !6
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %10)
  %12 = load i32, ptr @e, align 4, !tbaa !6
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %12)
  %14 = load i64, ptr @z, align 8, !tbaa !10
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i64 noundef %14)
  %16 = load i64, ptr @y, align 8, !tbaa !10
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i64 noundef %16)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!11 = !{!"long", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"short", !8, i64 0}
!14 = !{!8, !8, i64 0}
