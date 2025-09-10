; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-004.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2020-01-06-coverage-004.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.u = type { i16 }

@a = dso_local local_unnamed_addr global i32 0, align 4
@x = dso_local local_unnamed_addr global i64 0, align 8
@d = dso_local local_unnamed_addr global %union.u zeroinitializer, align 4
@y = dso_local local_unnamed_addr global i64 0, align 8
@h_call_argument_1 = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global ptr null, align 8
@b = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [8 x i8] c"a = %i\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"b = %i\0A\00", align 1
@.str.2 = private unnamed_addr constant [10 x i8] c"d.c = %u\0A\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"y = %li\0A\00", align 1
@.str.4 = private unnamed_addr constant [24 x i8] c"h_call_argument_1 = %i\0A\00", align 1
@.str.5 = private unnamed_addr constant [9 x i8] c"x = %li\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i64 0, 2) i64 @h(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %9, label %3

3:                                                ; preds = %1
  %4 = load i32, ptr @a, align 4, !tbaa !6
  %5 = sext i32 %4 to i64
  %6 = sdiv i64 2036854775807, %5
  %7 = icmp ne i64 %6, 0
  %8 = zext i1 %7 to i64
  br label %9

9:                                                ; preds = %3, %1
  %10 = phi i64 [ 0, %1 ], [ %8, %3 ]
  store i64 %10, ptr @x, align 8, !tbaa !10
  ret i64 %10
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define dso_local void @j() local_unnamed_addr #1 {
  %1 = load i16, ptr @d, align 4, !tbaa !12
  %2 = icmp eq i16 %1, 0
  br i1 %2, label %22, label %3

3:                                                ; preds = %0
  %4 = load ptr, ptr @f, align 8, !tbaa !13
  %5 = load i32, ptr @a, align 4
  %6 = sext i32 %5 to i64
  br label %7

7:                                                ; preds = %3, %16
  %8 = phi i16 [ %1, %3 ], [ %18, %16 ]
  %9 = load i64, ptr %4, align 8, !tbaa !10
  %10 = and i64 %9, 4294967295
  %11 = icmp eq i64 %10, 0
  br i1 %11, label %16, label %12

12:                                               ; preds = %7
  %13 = sdiv i64 2036854775807, %6
  %14 = icmp ne i64 %13, 0
  %15 = zext i1 %14 to i64
  br label %16

16:                                               ; preds = %7, %12
  %17 = phi i64 [ 0, %7 ], [ %15, %12 ]
  store i64 %17, ptr @x, align 8, !tbaa !10
  %18 = add i16 %8, 1
  %19 = icmp eq i16 %18, 0
  br i1 %19, label %20, label %7, !llvm.loop !16

20:                                               ; preds = %16
  %21 = trunc nuw nsw i64 %17 to i32
  store i32 %21, ptr @b, align 4, !tbaa !6
  store i16 0, ptr @d, align 4, !tbaa !12
  br label %22

22:                                               ; preds = %20, %0
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca i64, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store i64 0, ptr %1, align 8, !tbaa !10
  store i32 -7, ptr @h_call_argument_1, align 4, !tbaa !6
  store ptr %1, ptr @f, align 8, !tbaa !13
  store i32 251, ptr @a, align 4, !tbaa !6
  store i32 0, ptr @b, align 4, !tbaa !6
  store i16 0, ptr @d, align 4, !tbaa !12
  store i64 1, ptr @x, align 8, !tbaa !10
  store i64 1, ptr @y, align 8, !tbaa !10
  %2 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 251)
  %3 = load i32, ptr @b, align 4, !tbaa !6
  %4 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %3)
  %5 = load i16, ptr @d, align 4, !tbaa !12
  %6 = zext i16 %5 to i32
  %7 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6)
  %8 = load i64, ptr @y, align 8, !tbaa !10
  %9 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %8)
  %10 = load i32, ptr @h_call_argument_1, align 4, !tbaa !6
  %11 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %10)
  %12 = load i64, ptr @x, align 8, !tbaa !10
  %13 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i64 noundef %12)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

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
!12 = !{!8, !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 long", !15, i64 0}
!15 = !{!"any pointer", !8, i64 0}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.mustprogress"}
