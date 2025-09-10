; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strstr-asm.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strstr-asm.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"rld\00", align 1
@p = dso_local local_unnamed_addr global ptr @.str, align 8
@.str.1 = private unnamed_addr constant [12 x i8] c"hello world\00", align 1
@q = dso_local local_unnamed_addr global ptr @.str.1, align 8
@.str.2 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"h\00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.5 = private unnamed_addr constant [2 x i8] c"o\00", align 1
@.str.6 = private unnamed_addr constant [6 x i8] c"world\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = tail call ptr @my_strstr(ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.2) #3
  %2 = icmp eq ptr %1, @.str.1
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

4:                                                ; preds = %0
  %5 = tail call ptr @my_strstr(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @.str.1, i64 4), ptr noundef nonnull @.str.2) #3
  %6 = icmp eq ptr %5, getelementptr inbounds nuw (i8, ptr @.str.1, i64 4)
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #4
  unreachable

8:                                                ; preds = %4
  %9 = tail call ptr @my_strstr(ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.3) #3
  %10 = icmp eq ptr %9, @.str.1
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #4
  unreachable

12:                                               ; preds = %8
  %13 = tail call ptr @my_strstr(ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.4) #3
  %14 = icmp eq ptr %13, getelementptr inbounds nuw (i8, ptr @.str.1, i64 6)
  br i1 %14, label %16, label %15

15:                                               ; preds = %12
  tail call void @abort() #4
  unreachable

16:                                               ; preds = %12
  %17 = tail call ptr @my_strstr(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @.str.1, i64 6), ptr noundef nonnull @.str.5) #3
  %18 = icmp eq ptr %17, getelementptr inbounds nuw (i8, ptr @.str.1, i64 7)
  br i1 %18, label %20, label %19

19:                                               ; preds = %16
  tail call void @abort() #4
  unreachable

20:                                               ; preds = %16
  %21 = tail call ptr @my_strstr(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @.str.1, i64 1), ptr noundef nonnull @.str.6) #3
  %22 = icmp eq ptr %21, getelementptr inbounds nuw (i8, ptr @.str.1, i64 6)
  br i1 %22, label %24, label %23

23:                                               ; preds = %20
  tail call void @abort() #4
  unreachable

24:                                               ; preds = %20
  %25 = load ptr, ptr @p, align 8, !tbaa !6
  %26 = tail call ptr @my_strstr(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @.str.1, i64 2), ptr noundef %25) #3
  %27 = icmp eq ptr %26, getelementptr inbounds nuw (i8, ptr @.str.1, i64 8)
  br i1 %27, label %29, label %28

28:                                               ; preds = %24
  tail call void @abort() #4
  unreachable

29:                                               ; preds = %24
  %30 = load ptr, ptr @q, align 8, !tbaa !6
  %31 = tail call ptr @my_strstr(ptr noundef %30, ptr noundef nonnull @.str.2) #3
  %32 = load ptr, ptr @q, align 8, !tbaa !6
  %33 = icmp eq ptr %31, %32
  br i1 %33, label %35, label %34

34:                                               ; preds = %29
  tail call void @abort() #4
  unreachable

35:                                               ; preds = %29
  %36 = getelementptr inbounds nuw i8, ptr %32, i64 1
  %37 = tail call ptr @my_strstr(ptr noundef nonnull %36, ptr noundef nonnull @.str.5) #3
  %38 = load ptr, ptr @q, align 8, !tbaa !6
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 4
  %40 = icmp eq ptr %37, %39
  br i1 %40, label %42, label %41

41:                                               ; preds = %35
  tail call void @abort() #4
  unreachable

42:                                               ; preds = %35
  ret void
}

; Function Attrs: nounwind
declare ptr @my_strstr(ptr noundef, ptr noundef) local_unnamed_addr #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

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
