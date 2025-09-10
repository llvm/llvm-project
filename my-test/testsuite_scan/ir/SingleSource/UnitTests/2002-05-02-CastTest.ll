; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2002-05-02-CastTest.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2002-05-02-CastTest.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [23 x i8] c"ubc0 = '%c'\09   [0x%x]\0A\00", align 1
@.str.1 = private unnamed_addr constant [21 x i8] c"ubs0 = %u\09   [0x%x]\0A\00", align 1
@.str.2 = private unnamed_addr constant [21 x i8] c"ubs1 = %u\09   [0x%x]\0A\00", align 1
@.str.3 = private unnamed_addr constant [21 x i8] c"bs0  = %d\09   [0x%x]\0A\00", align 1
@.str.4 = private unnamed_addr constant [21 x i8] c"bs1  = %d\09   [0x%x]\0A\00", align 1
@.str.5 = private unnamed_addr constant [23 x i8] c"c1   = '%c'\09   [0x%x]\0A\00", align 1
@.str.6 = private unnamed_addr constant [21 x i8] c"s1   = %d\09   [0x%x]\0A\00", align 1
@.str.7 = private unnamed_addr constant [23 x i8] c"uc2  = '%c'\09   [0x%x]\0A\00", align 1
@.str.8 = private unnamed_addr constant [21 x i8] c"us2  = %u\09   [0x%x]\0A\00", align 1
@.str.9 = private unnamed_addr constant [23 x i8] c"ic3  = '%c'\09   [0x%x]\0A\00", align 1
@.str.10 = private unnamed_addr constant [21 x i8] c"is3  = %d\09   [0x%x]\0A\00", align 1
@.str.11 = private unnamed_addr constant [21 x i8] c"is4  = %d\09   [0x%x]\0A\00", align 1
@.str.12 = private unnamed_addr constant [21 x i8] c"is5  = %d\09   [0x%x]\0A\00", align 1
@.str.13 = private unnamed_addr constant [23 x i8] c"uic4 = '%c'\09   [0x%x]\0A\00", align 1
@.str.14 = private unnamed_addr constant [19 x i8] c"uis4 = %u  [0x%x]\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %16

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #3
  %8 = trunc i64 %7 to i32
  %9 = and i32 %8, 255
  %10 = icmp eq i32 %0, 2
  br i1 %10, label %16, label %11

11:                                               ; preds = %4
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load ptr, ptr %12, align 8, !tbaa !6
  %14 = tail call i64 @strtol(ptr noundef nonnull captures(none) %13, ptr noundef null, i32 noundef 10) #3
  %15 = trunc i64 %14 to i16
  br label %16

16:                                               ; preds = %2, %4, %11
  %17 = phi i32 [ %9, %11 ], [ %9, %4 ], [ 100, %2 ]
  %18 = phi i16 [ %15, %11 ], [ -769, %4 ], [ -769, %2 ]
  %19 = zext i16 %18 to i32
  %20 = and i16 %18, 255
  %21 = zext nneg i16 %20 to i32
  %22 = add nuw nsw i32 %21, 1
  %23 = sext i16 %18 to i32
  %24 = shl i32 %19, 24
  %25 = ashr exact i32 %24, 24
  %26 = add nsw i32 %25, 1
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %17, i32 noundef %17)
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %21, i32 noundef %21)
  %29 = and i32 %22, 255
  %30 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %29, i32 noundef %29)
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %25, i32 noundef %25)
  %32 = shl i32 %22, 24
  %33 = ashr exact i32 %32, 24
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %33, i32 noundef %33)
  %35 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %17, i32 noundef %17)
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef %23, i32 noundef %23)
  %37 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %17, i32 noundef %17)
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef %19, i32 noundef %19)
  %39 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef %17, i32 noundef %17)
  %40 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %23, i32 noundef %23)
  %41 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %22, i32 noundef %22)
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef %26, i32 noundef %26)
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef %17, i32 noundef %17)
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef %23, i32 noundef %23)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
