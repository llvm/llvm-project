; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-009.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-009.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@d = dso_local global i8 8, align 4
@e = dso_local local_unnamed_addr constant ptr @d, align 8
@c = dso_local local_unnamed_addr global i16 0, align 4
@g = dso_local local_unnamed_addr global i16 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global ptr null, align 8
@.str = private unnamed_addr constant [8 x i8] c"a = %u\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"b = %u\0A\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"c = %i\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"d = %u\0A\00", align 1
@.str.4 = private unnamed_addr constant [8 x i8] c"g = %i\0A\00", align 1
@.str.5 = private unnamed_addr constant [8 x i8] c"h = %u\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @i() local_unnamed_addr #0 {
  %1 = load i16, ptr @c, align 4, !tbaa !6
  %2 = icmp eq i16 %1, 0
  br i1 %2, label %59, label %3

3:                                                ; preds = %0
  %4 = load i8, ptr @d, align 4
  %5 = load ptr, ptr @f, align 8, !tbaa !10
  br label %6

6:                                                ; preds = %3, %6
  %7 = phi i16 [ %1, %3 ], [ %56, %6 ]
  %8 = phi i8 [ %4, %3 ], [ %55, %6 ]
  %9 = load i32, ptr @b, align 4, !tbaa !13
  %10 = load i32, ptr @a, align 4, !tbaa !13
  %11 = xor i32 %10, %9
  %12 = icmp ult i32 %11, %9
  %13 = zext i1 %12 to i8
  %14 = xor i8 %8, %13
  store i32 0, ptr %5, align 4, !tbaa !13
  %15 = load i32, ptr @b, align 4, !tbaa !13
  %16 = load i32, ptr @a, align 4, !tbaa !13
  %17 = xor i32 %16, %15
  %18 = icmp ult i32 %17, %15
  %19 = zext i1 %18 to i8
  %20 = xor i8 %14, %19
  %21 = xor i32 %16, %15
  %22 = icmp ult i32 %21, %15
  %23 = zext i1 %22 to i8
  %24 = xor i8 %20, %23
  %25 = load i32, ptr @b, align 4, !tbaa !13
  %26 = load i32, ptr @a, align 4, !tbaa !13
  %27 = xor i32 %26, %25
  %28 = icmp ult i32 %27, %25
  %29 = zext i1 %28 to i8
  %30 = xor i8 %24, %29
  %31 = xor i32 %26, %25
  %32 = icmp ult i32 %31, %25
  %33 = zext i1 %32 to i8
  %34 = xor i8 %30, %33
  %35 = load i32, ptr @b, align 4, !tbaa !13
  %36 = load i32, ptr @a, align 4, !tbaa !13
  %37 = xor i32 %36, %35
  %38 = icmp ult i32 %37, %35
  %39 = zext i1 %38 to i8
  %40 = xor i8 %34, %39
  %41 = xor i32 %36, %35
  %42 = icmp ult i32 %41, %35
  %43 = zext i1 %42 to i8
  %44 = xor i8 %40, %43
  %45 = load i32, ptr @b, align 4, !tbaa !13
  %46 = load i32, ptr @a, align 4, !tbaa !13
  %47 = xor i32 %46, %45
  %48 = icmp ult i32 %47, %45
  %49 = zext i1 %48 to i8
  %50 = xor i8 %44, %49
  %51 = xor i32 %46, %45
  %52 = icmp ult i32 %51, %45
  %53 = zext i1 %52 to i32
  store i32 %53, ptr @h, align 4, !tbaa !13
  %54 = zext i1 %52 to i8
  %55 = xor i8 %50, %54
  store i32 0, ptr %5, align 4, !tbaa !13
  %56 = add i16 %7, -1
  %57 = icmp eq i16 %56, 0
  br i1 %57, label %58, label %6, !llvm.loop !15

58:                                               ; preds = %6
  store i16 9, ptr @g, align 4, !tbaa !6
  store i8 %55, ptr @d, align 4, !tbaa !17
  store i16 0, ptr @c, align 4, !tbaa !6
  br label %59

59:                                               ; preds = %58, %0
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  store i32 -2130706432, ptr %1, align 4, !tbaa !13
  store ptr %1, ptr @f, align 8, !tbaa !10
  store i16 -32767, ptr @g, align 4, !tbaa !6
  store i16 -32767, ptr @c, align 4, !tbaa !6
  store i32 -1, ptr @h, align 4, !tbaa !13
  store i32 1, ptr @b, align 4, !tbaa !13
  store i32 0, ptr @a, align 4, !tbaa !13
  call void @i()
  %2 = load i32, ptr @a, align 4, !tbaa !13
  %3 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %2)
  %4 = load i32, ptr @b, align 4, !tbaa !13
  %5 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %4)
  %6 = load i16, ptr @c, align 4, !tbaa !6
  %7 = sext i16 %6 to i32
  %8 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %7)
  %9 = load i8, ptr @d, align 4, !tbaa !17
  %10 = zext i8 %9 to i32
  %11 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %10)
  %12 = load i16, ptr @g, align 4, !tbaa !6
  %13 = sext i16 %12 to i32
  %14 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %13)
  %15 = load i32, ptr @h, align 4, !tbaa !13
  %16 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }

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
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !8, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!8, !8, i64 0}
