; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-005.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-005.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local global i64 0, align 8
@r = dso_local local_unnamed_addr global ptr null, align 8
@s = dso_local local_unnamed_addr global ptr null, align 8
@p = dso_local local_unnamed_addr global ptr null, align 8
@e = dso_local global i16 0, align 4
@t = dso_local local_unnamed_addr global ptr null, align 8
@q = dso_local local_unnamed_addr global ptr null, align 8
@c = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [8 x i8] c"a = %u\0A\00", align 1
@.str.1 = private unnamed_addr constant [9 x i8] c"b = %lu\0A\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"c = %u\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"d = %u\0A\00", align 1
@.str.4 = private unnamed_addr constant [8 x i8] c"e = %i\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @h() local_unnamed_addr #0 {
  store ptr @b, ptr @r, align 8, !tbaa !6
  %1 = load i32, ptr @c, align 4, !tbaa !11
  %2 = load i32, ptr @d, align 4, !tbaa !11
  %3 = icmp eq i32 %2, 0
  %4 = load i16, ptr @e, align 4, !tbaa !13
  %5 = and i32 %1, 6
  %6 = zext nneg i32 %5 to i64
  store ptr @b, ptr @s, align 8, !tbaa !6
  store ptr @b, ptr @p, align 8, !tbaa !6
  store ptr @e, ptr @t, align 8, !tbaa !15
  store ptr @e, ptr @q, align 8, !tbaa !15
  br i1 %3, label %9, label %7

7:                                                ; preds = %0
  store i16 0, ptr @e, align 4, !tbaa !13
  store i64 %6, ptr @b, align 8, !tbaa !17
  store i32 %2, ptr @a, align 4, !tbaa !11
  br label %8

8:                                                ; preds = %8, %7
  br label %8

9:                                                ; preds = %0
  %10 = add i16 %4, -1
  store i16 %10, ptr @e, align 4, !tbaa !13
  store i64 %6, ptr @b, align 8, !tbaa !17
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @g() local_unnamed_addr #0 {
  %1 = load i32, ptr @c, align 4, !tbaa !11
  %2 = zext i32 %1 to i64
  %3 = load i32, ptr @d, align 4, !tbaa !11
  %4 = icmp eq i32 %3, 0
  %5 = load i16, ptr @e, align 4, !tbaa !13
  %6 = load i64, ptr @b, align 8, !tbaa !17
  %7 = and i64 %6, %2
  store ptr @b, ptr @s, align 8, !tbaa !6
  store ptr @b, ptr @p, align 8, !tbaa !6
  store ptr @e, ptr @t, align 8, !tbaa !15
  store ptr @e, ptr @q, align 8, !tbaa !15
  br i1 %4, label %10, label %8

8:                                                ; preds = %0
  store i16 0, ptr @e, align 4, !tbaa !13
  store i64 %7, ptr @b, align 8, !tbaa !17
  store i32 %3, ptr @a, align 4, !tbaa !11
  br label %9

9:                                                ; preds = %9, %8
  br label %9

10:                                               ; preds = %0
  %11 = add i16 %5, -1
  store i16 %11, ptr @e, align 4, !tbaa !13
  store i64 %7, ptr @b, align 8, !tbaa !17
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i32 0, ptr @d, align 4, !tbaa !11
  store i32 -6, ptr @c, align 4, !tbaa !11
  store i32 16777101, ptr @a, align 4, !tbaa !11
  store ptr @b, ptr @r, align 8, !tbaa !6
  store ptr @b, ptr @s, align 8, !tbaa !6
  store ptr @b, ptr @p, align 8, !tbaa !6
  store ptr @e, ptr @t, align 8, !tbaa !15
  store ptr @e, ptr @q, align 8, !tbaa !15
  store i16 -9, ptr @e, align 4, !tbaa !13
  store i64 2, ptr @b, align 8, !tbaa !17
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 16777101)
  %2 = load i64, ptr @b, align 8, !tbaa !17
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i64 noundef %2)
  %4 = load i32, ptr @c, align 4, !tbaa !11
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %4)
  %6 = load i32, ptr @d, align 4, !tbaa !11
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %6)
  %8 = load i16, ptr @e, align 4, !tbaa !13
  %9 = sext i16 %8 to i32
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %9)
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
!7 = !{!"p1 long", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"short", !9, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"p1 short", !8, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"long", !9, i64 0}
