; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcpy.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcpy.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@p = dso_local global [32 x i8] zeroinitializer, align 1
@.str = private unnamed_addr constant [6 x i8] c"abcde\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"vwxyz\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"wxyz\00", align 1
@.str.4 = private unnamed_addr constant [6 x i8] c"a\00cde\00", align 1
@.str.5 = private unnamed_addr constant [6 x i8] c"fghij\00", align 1
@.str.6 = private unnamed_addr constant [9 x i8] c"a\00cfghij\00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c"ABCDE\00", align 1
@.str.9 = private unnamed_addr constant [5 x i8] c"WXyz\00", align 1
@.str.10 = private unnamed_addr constant [6 x i8] c"A\00CDE\00", align 1
@.str.12 = private unnamed_addr constant [9 x i8] c"A\00CFGHIj\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) @p, ptr noundef nonnull align 1 dereferenceable(6) @.str, i64 6, i1 false) #4
  %1 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str, i64 6)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

4:                                                ; preds = %0
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) getelementptr inbounds nuw (i8, ptr @p, i64 16), ptr noundef nonnull align 1 dereferenceable(5) getelementptr inbounds nuw (i8, ptr @.str.1, i64 1), i64 5, i1 false) #4
  %5 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(5) getelementptr inbounds nuw (i8, ptr @p, i64 16), ptr noundef nonnull dereferenceable(5) @.str.2, i64 5)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %4
  store i8 0, ptr getelementptr inbounds nuw (i8, ptr @p, i64 1), align 1
  %9 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.4, i64 6)
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #5
  unreachable

12:                                               ; preds = %8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) getelementptr inbounds nuw (i8, ptr @p, i64 3), ptr noundef nonnull align 1 dereferenceable(6) @.str.5, i64 6, i1 false) #4
  %13 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(9) @p, ptr noundef nonnull dereferenceable(9) @.str.6, i64 9)
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %12
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %12
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) @p, ptr noundef nonnull align 1 dereferenceable(6) @.str.7, i64 6, i1 false)
  %17 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.7, i64 6)
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %16
  tail call void @abort() #5
  unreachable

20:                                               ; preds = %16
  store i16 22615, ptr getelementptr inbounds nuw (i8, ptr @p, i64 16), align 1
  %21 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(5) getelementptr inbounds nuw (i8, ptr @p, i64 16), ptr noundef nonnull dereferenceable(5) @.str.9, i64 5)
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %24, label %23

23:                                               ; preds = %20
  tail call void @abort() #5
  unreachable

24:                                               ; preds = %20
  store i8 0, ptr getelementptr inbounds nuw (i8, ptr @p, i64 1), align 1
  %25 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.10, i64 6)
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %28, label %27

27:                                               ; preds = %24
  tail call void @abort() #5
  unreachable

28:                                               ; preds = %24
  store i32 1229473606, ptr getelementptr inbounds nuw (i8, ptr @p, i64 3), align 1
  %29 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(9) @p, ptr noundef nonnull dereferenceable(9) @.str.12, i64 9)
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %32, label %31

31:                                               ; preds = %28
  tail call void @abort() #5
  unreachable

32:                                               ; preds = %28
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) @p, ptr noundef nonnull align 1 dereferenceable(6) @.str, i64 6, i1 false) #4
  %33 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str, i64 6)
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %36, label %35

35:                                               ; preds = %32
  tail call void @abort() #5
  unreachable

36:                                               ; preds = %32
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) @p, ptr noundef nonnull align 1 dereferenceable(6) @.str.7, i64 6, i1 false)
  %37 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.7, i64 6)
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %36
  tail call void @abort() #5
  unreachable

40:                                               ; preds = %36
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
