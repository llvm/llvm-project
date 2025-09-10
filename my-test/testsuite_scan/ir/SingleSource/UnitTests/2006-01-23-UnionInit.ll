; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2006-01-23-UnionInit.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2006-01-23-UnionInit.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.task_struct = type { [16 x i8] }
%struct.anon = type { i8, i8 }
%union.bt4 = type { %struct.at4 }
%struct.at4 = type { i64 }
%struct.anon.0 = type { i8, i8 }
%union.anon.3 = type { %struct.dvd_send_key }
%struct.dvd_send_key = type { i8, [5 x i8], [2 x i8] }
%struct.foo2 = type <{ i16, [2 x i8], i64, i16, [2 x i8] }>

@init_task_union = dso_local global { %struct.task_struct, [16368 x i8] } { %struct.task_struct { [16 x i8] c"swapper\00\00\00\00\00\00\00\00\00" }, [16368 x i8] zeroinitializer }, align 8
@fixed_tl = dso_local local_unnamed_addr global [1 x { { %struct.anon, [2 x i8] }, i32 }] [{ { %struct.anon, [2 x i8] }, i32 } { { %struct.anon, [2 x i8] } { %struct.anon { i8 96, i8 7 }, [2 x i8] zeroinitializer }, i32 256 }], align 4
@bkv4 = dso_local local_unnamed_addr global %union.bt4 { %struct.at4 { i64 61172160 } }, align 8
@cav1 = dso_local local_unnamed_addr global { i8, i8, [1 x i8], i8 } { i8 100, i8 1, [1 x i8] c"\02", i8 0 }, align 4
@V = dso_local local_unnamed_addr global { %struct.anon.0, [2 x i8] } { %struct.anon.0 { i8 1, i8 2 }, [2 x i8] zeroinitializer }, align 4
@nd = dso_local global { [64 x i8] } { [64 x i8] c"relname\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00" }, align 4
@hobbit = dso_local local_unnamed_addr global { i32, [3 x i32] } { i32 5, [3 x i32] [i32 1, i32 2, i32 3] }, align 4
@hobbit2 = dso_local global { i32, [3 x i32] } { i32 5, [3 x i32] [i32 1, i32 2, i32 3] }, align 4
@data = dso_local local_unnamed_addr global { i32, [3 x i32] } { i32 0, [3 x i32] [i32 1, i32 2, i32 3] }, align 4
@s = dso_local global <{ ptr, [19 x i8] }> <{ ptr null, [19 x i8] c"   xyzkasjdlf     \00" }>, align 8
@ai = dso_local global %union.anon.3 { %struct.dvd_send_key { i8 3, [5 x i8] c"foo\00\00", [2 x i8] zeroinitializer } }, align 4
@foo2 = dso_local local_unnamed_addr global %struct.foo2 <{ i16 23122, [2 x i8] zeroinitializer, i64 -12312731, i16 -312, [2 x i8] zeroinitializer }>, align 4
@.str.2 = private unnamed_addr constant [11 x i8] c"PR156: %s\0A\00", align 1
@.str.3 = private unnamed_addr constant [21 x i8] c"PR295/PR568: %d, %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [23 x i8] c"PR574: %d, %d, %d, %d\0A\00", align 1
@.str.5 = private unnamed_addr constant [19 x i8] c"PR162: %d, %d, %d\0A\00", align 1
@.str.6 = private unnamed_addr constant [15 x i8] c"PR650: %s, %d\0A\00", align 1
@.str.7 = private unnamed_addr constant [23 x i8] c"PR199: %d, %d, %d, %d\0A\00", align 1
@.str.8 = private unnamed_addr constant [23 x i8] c"PR431: %d, %d, %d, %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [18 x i8] c"PR654: %ld, '%s'\0A\00", align 1
@.str.10 = private unnamed_addr constant [17 x i8] c"PR323: %d, '%s'\0A\00", align 1
@.str.11 = private unnamed_addr constant [11 x i8] c"PR627: %d\0A\00", align 1
@.str.12 = private unnamed_addr constant [22 x i8] c"PR684: %d, %d, %d %d\0A\00", align 1
@.str.13 = private unnamed_addr constant [28 x i8] c"rdar://6828787: %d, %d, %d\0A\00", align 1
@str = private unnamed_addr constant [19 x i8] c"returning raw_lock\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef nonnull ptr @foo() local_unnamed_addr #0 {
  ret ptr @hobbit2
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @one_raw_spinlock() local_unnamed_addr #1 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @ucast_test() local_unnamed_addr #0 {
  ret [2 x i64] [i64 8589934593, i64 0]
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, ptr noundef nonnull @init_task_union)
  %2 = load i32, ptr @fixed_tl, align 4, !tbaa !6
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @fixed_tl, i64 4), align 4, !tbaa !9
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %2, i32 noundef %3)
  %5 = load i64, ptr @bkv4, align 8, !tbaa !6
  %6 = trunc i64 %5 to i32
  %7 = load i16, ptr @cav1, align 4
  %8 = shl i16 %7, 12
  %9 = ashr exact i16 %8, 12
  %10 = sext i16 %9 to i32
  %11 = shl i16 %7, 6
  %12 = ashr i16 %11, 10
  %13 = sext i16 %12 to i32
  %14 = load i8, ptr getelementptr inbounds nuw (i8, ptr @cav1, i64 2), align 2, !tbaa !6
  %15 = zext i8 %14 to i32
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %6, i32 noundef %10, i32 noundef %13, i32 noundef %15)
  %17 = load i8, ptr @V, align 4, !tbaa !6
  %18 = zext i8 %17 to i32
  %19 = load i8, ptr getelementptr inbounds nuw (i8, ptr @V, i64 1), align 1, !tbaa !6
  %20 = zext i8 %19 to i32
  %21 = load i32, ptr @V, align 4, !tbaa !6
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %18, i32 noundef %20, i32 noundef %21)
  %23 = load i32, ptr @nd, align 4, !tbaa !6
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, ptr noundef nonnull @nd, i32 noundef %23)
  %25 = load i32, ptr @hobbit, align 4, !tbaa !12
  %26 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hobbit, i64 4), align 4, !tbaa !12
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hobbit, i64 8), align 4, !tbaa !12
  %28 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hobbit, i64 12), align 4, !tbaa !12
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %25, i32 noundef %26, i32 noundef %27, i32 noundef %28)
  %30 = load i32, ptr @hobbit2, align 4, !tbaa !12
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hobbit2, i64 4), align 4, !tbaa !12
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hobbit2, i64 8), align 4, !tbaa !12
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @hobbit2, i64 12), align 4, !tbaa !12
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %30, i32 noundef %31, i32 noundef %32, i32 noundef %33)
  %35 = load i32, ptr @data, align 4, !tbaa !12
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @data, i64 4), align 4, !tbaa !12
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @data, i64 8), align 4, !tbaa !12
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @data, i64 12), align 4, !tbaa !12
  %39 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef %35, i32 noundef %36, i32 noundef %37, i32 noundef %38)
  %40 = load ptr, ptr @s, align 8, !tbaa !13
  %41 = ptrtoint ptr %40 to i64
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i64 noundef %41, ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @s, i64 8))
  %43 = load i8, ptr @ai, align 4
  %44 = and i8 %43, 3
  %45 = zext nneg i8 %44 to i32
  %46 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %45, ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @ai, i64 1))
  %47 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %48 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef 0)
  %49 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef 1, i32 noundef 2, i32 noundef 0, i32 noundef 1)
  %50 = load i16, ptr @foo2, align 4, !tbaa !16
  %51 = sext i16 %50 to i32
  %52 = load i64, ptr getelementptr inbounds nuw (i8, ptr @foo2, i64 4), align 4, !tbaa !20
  %53 = trunc i64 %52 to i32
  %54 = load i16, ptr getelementptr inbounds nuw (i8, ptr @foo2, i64 12), align 4, !tbaa !21
  %55 = sext i16 %54 to i32
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef %51, i32 noundef %53, i32 noundef %55)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !11, i64 4}
!10 = !{!"inflate_huft_s", !7, i64 0, !11, i64 4}
!11 = !{!"int", !7, i64 0}
!12 = !{!11, !11, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 omnipotent char", !15, i64 0}
!15 = !{!"any pointer", !7, i64 0}
!16 = !{!17, !18, i64 0}
!17 = !{!"foo2", !18, i64 0, !19, i64 4, !18, i64 12}
!18 = !{!"short", !7, i64 0}
!19 = !{!"long", !7, i64 0}
!20 = !{!17, !19, i64 4}
!21 = !{!17, !18, i64 12}
