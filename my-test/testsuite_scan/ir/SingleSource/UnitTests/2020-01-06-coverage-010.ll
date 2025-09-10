; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-010.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-010.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v = dso_local local_unnamed_addr global i32 0, align 4
@w = dso_local local_unnamed_addr global i32 0, align 4
@z = dso_local local_unnamed_addr global i32 0, align 4
@y = dso_local local_unnamed_addr global i32 0, align 4
@p = dso_local local_unnamed_addr global ptr null, align 8
@x = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local global i32 0, align 4
@c = dso_local global i8 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global ptr null, align 8
@.str = private unnamed_addr constant [8 x i8] c"b = %i\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"c = %i\0A\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"d = %i\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"x = %i\0A\00", align 1
@.str.4 = private unnamed_addr constant [8 x i8] c"e = %i\0A\00", align 1
@.str.5 = private unnamed_addr constant [8 x i8] c"y = %i\0A\00", align 1
@.str.6 = private unnamed_addr constant [8 x i8] c"z = %i\0A\00", align 1
@.str.7 = private unnamed_addr constant [8 x i8] c"v = %i\0A\00", align 1
@.str.8 = private unnamed_addr constant [8 x i8] c"f = %i\0A\00", align 1
@.str.9 = private unnamed_addr constant [8 x i8] c"w = %i\0A\00", align 1

; Function Attrs: nofree norecurse nounwind memory(readwrite, argmem: write) uwtable
define dso_local void @k() local_unnamed_addr #0 {
  store i32 5, ptr @x, align 4, !tbaa !6
  %1 = load i32, ptr @b, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %49, label %3

3:                                                ; preds = %0
  %4 = load ptr, ptr @a, align 8, !tbaa !10
  br label %5

5:                                                ; preds = %43, %3
  %6 = phi i32 [ 5, %3 ], [ %45, %43 ]
  %7 = load volatile i32, ptr @e, align 4, !tbaa !6
  switch i32 %6, label %8 [
    i32 0, label %13
    i32 -1, label %13
  ]

8:                                                ; preds = %5
  %9 = load volatile i32, ptr @e, align 4, !tbaa !6
  %10 = load volatile i32, ptr @e, align 4, !tbaa !6
  %11 = add i32 %6, 3
  %12 = icmp ult i32 %11, 2
  br i1 %12, label %21, label %16

13:                                               ; preds = %5, %5
  %14 = load volatile i32, ptr @e, align 4, !tbaa !6
  %15 = load volatile i32, ptr @e, align 4, !tbaa !6
  br label %21

16:                                               ; preds = %8
  %17 = load volatile i32, ptr @e, align 4, !tbaa !6
  %18 = load volatile i32, ptr @e, align 4, !tbaa !6
  %19 = add i32 %6, 5
  %20 = icmp ult i32 %19, 2
  br i1 %20, label %29, label %24

21:                                               ; preds = %8, %13
  %22 = load volatile i32, ptr @e, align 4, !tbaa !6
  %23 = load volatile i32, ptr @e, align 4, !tbaa !6
  br label %29

24:                                               ; preds = %16
  %25 = load volatile i32, ptr @e, align 4, !tbaa !6
  %26 = load volatile i32, ptr @e, align 4, !tbaa !6
  %27 = add i32 %6, 7
  %28 = icmp ult i32 %27, 2
  br i1 %28, label %35, label %32

29:                                               ; preds = %21, %16
  %30 = load volatile i32, ptr @e, align 4, !tbaa !6
  store i8 0, ptr @c, align 4, !tbaa !13
  %31 = load volatile i32, ptr @e, align 4, !tbaa !6
  br label %35

32:                                               ; preds = %24
  %33 = load volatile i32, ptr @e, align 4, !tbaa !6
  %34 = icmp eq i32 %6, -8
  br i1 %34, label %40, label %37

35:                                               ; preds = %29, %24
  %36 = load volatile i32, ptr @e, align 4, !tbaa !6
  br label %40

37:                                               ; preds = %32
  store i8 0, ptr @c, align 4, !tbaa !13
  %38 = load volatile i32, ptr @e, align 4, !tbaa !6
  %39 = icmp eq i32 %6, -9
  br i1 %39, label %43, label %42

40:                                               ; preds = %35, %32
  %41 = load volatile i32, ptr @e, align 4, !tbaa !6
  br label %42

42:                                               ; preds = %40, %37
  store i8 0, ptr @c, align 4, !tbaa !13
  br label %43

43:                                               ; preds = %42, %37
  %44 = load volatile i32, ptr @e, align 4, !tbaa !6
  %45 = add i32 %6, 10
  store i32 10, ptr @f, align 4, !tbaa !6
  store i32 0, ptr @y, align 4, !tbaa !6
  store i32 %44, ptr @z, align 4, !tbaa !6
  store i32 0, ptr %4, align 4, !tbaa !6
  %46 = load i32, ptr @b, align 4, !tbaa !6
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %48, label %5, !llvm.loop !14

48:                                               ; preds = %43
  store ptr @c, ptr @p, align 8, !tbaa !16
  br label %49

49:                                               ; preds = %0, %48
  %50 = load i32, ptr @d, align 4, !tbaa !6
  store i32 %50, ptr @w, align 4, !tbaa !6
  store i32 %50, ptr @v, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca i8, align 4
  %2 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  store i32 0, ptr %2, align 4, !tbaa !6
  store i8 5, ptr %1, align 4, !tbaa !13
  store i32 0, ptr @v, align 4, !tbaa !6
  store i32 0, ptr @w, align 4, !tbaa !6
  store i32 0, ptr @z, align 4, !tbaa !6
  store i32 0, ptr @y, align 4, !tbaa !6
  store ptr %1, ptr @p, align 8, !tbaa !16
  store i32 0, ptr @x, align 4, !tbaa !6
  store volatile i32 0, ptr @e, align 4, !tbaa !6
  store i8 0, ptr @c, align 4, !tbaa !13
  store i32 0, ptr @f, align 4, !tbaa !6
  store i32 0, ptr @d, align 4, !tbaa !6
  store i32 0, ptr @b, align 4, !tbaa !6
  store ptr %2, ptr @a, align 8, !tbaa !10
  call void @k()
  %3 = load i32, ptr @b, align 4, !tbaa !6
  %4 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %3)
  %5 = load i8, ptr @c, align 4, !tbaa !13
  %6 = sext i8 %5 to i32
  %7 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %6)
  %8 = load i32, ptr @d, align 4, !tbaa !6
  %9 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %8)
  %10 = load i32, ptr @x, align 4, !tbaa !6
  %11 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %10)
  %12 = load volatile i32, ptr @e, align 4, !tbaa !6
  %13 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %12)
  %14 = load i32, ptr @y, align 4, !tbaa !6
  %15 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %14)
  %16 = load i32, ptr @z, align 4, !tbaa !6
  %17 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef %16)
  %18 = load i32, ptr @v, align 4, !tbaa !6
  %19 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %18)
  %20 = load i32, ptr @f, align 4, !tbaa !6
  %21 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef %20)
  %22 = load i32, ptr @w, align 4, !tbaa !6
  %23 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

attributes #0 = { nofree norecurse nounwind memory(readwrite, argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!8, !8, i64 0}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 omnipotent char", !12, i64 0}
