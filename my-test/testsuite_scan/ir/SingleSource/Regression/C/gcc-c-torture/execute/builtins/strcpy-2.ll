; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcpy-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcpy-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [8 x i8] c"abcdefg\00", align 1
@i = dso_local local_unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [8 x i8] c"bcdefgh\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"cdefghi\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"defghij\00", align 1
@buf = dso_local global [32 x i8] zeroinitializer, align 8
@p = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define dso_local nonnull ptr @test() local_unnamed_addr #0 {
  %1 = load i32, ptr @i, align 4, !tbaa !6
  %2 = add nsw i32 %1, 1
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %11, label %4

4:                                                ; preds = %0
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %11, label %6

6:                                                ; preds = %4
  switch i32 %1, label %7 [
    i32 1, label %16
    i32 2, label %15
  ]

7:                                                ; preds = %6
  %8 = icmp ugt i32 %1, -3
  %9 = select i1 %8, ptr @.str.3, ptr @.str
  %10 = icmp eq i32 %2, 2
  br i1 %10, label %16, label %11

11:                                               ; preds = %4, %0, %7
  %12 = phi ptr [ %9, %7 ], [ @.str.2, %0 ], [ @.str.3, %4 ]
  %13 = icmp eq i32 %1, 0
  %14 = select i1 %13, ptr @.str.3, ptr %12
  br label %16

15:                                               ; preds = %6
  br label %16

16:                                               ; preds = %6, %15, %11, %7
  %17 = phi ptr [ @.str.2, %7 ], [ %14, %11 ], [ @.str.2, %6 ], [ @.str.1, %15 ]
  %18 = load i64, ptr %17, align 1
  store i64 %18, ptr @buf, align 8
  store ptr @buf, ptr @p, align 8, !tbaa !10
  store i64 %18, ptr getelementptr inbounds nuw (i8, ptr @buf, i64 16), align 8
  ret ptr getelementptr inbounds nuw (i8, ptr @buf, i64 16)
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #1 {
  %1 = tail call ptr @test()
  %2 = icmp ne ptr %1, getelementptr inbounds nuw (i8, ptr @buf, i64 16)
  %3 = load ptr, ptr @p, align 8
  %4 = icmp ne ptr %3, @buf
  %5 = select i1 %2, i1 true, i1 %4
  br i1 %5, label %6, label %7

6:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

7:                                                ; preds = %0
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
