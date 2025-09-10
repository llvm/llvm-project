; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strpcpy.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strpcpy.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@s1 = dso_local local_unnamed_addr constant [4 x i8] c"123\00", align 1
@p = dso_local global [32 x i8] zeroinitializer, align 8
@.str = private unnamed_addr constant [5 x i8] c"defg\00", align 1
@s2 = dso_local local_unnamed_addr global ptr @.str, align 8
@.str.1 = private unnamed_addr constant [4 x i8] c"FGH\00", align 1
@s3 = dso_local local_unnamed_addr global ptr @.str.1, align 8
@l1 = dso_local local_unnamed_addr global i64 1, align 8
@inside_main = external local_unnamed_addr global i32, align 4
@.str.2 = private unnamed_addr constant [6 x i8] c"abcde\00", align 1
@.str.3 = private unnamed_addr constant [6 x i8] c"vwxyz\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"wxyz\00", align 1
@.str.6 = private unnamed_addr constant [6 x i8] c"a\00cde\00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c"fghij\00", align 1
@.str.8 = private unnamed_addr constant [9 x i8] c"a\00cfghij\00", align 1
@.str.9 = private unnamed_addr constant [3 x i8] c"23\00", align 1
@.str.10 = private unnamed_addr constant [6 x i8] c"z\0023\00\00", align 1
@.str.13 = private unnamed_addr constant [8 x i8] c"ABCDEFG\00", align 1
@.str.14 = private unnamed_addr constant [7 x i8] c"abcFGH\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  store i32 0, ptr @inside_main, align 4, !tbaa !6
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(6) @p, ptr noundef nonnull align 1 dereferenceable(6) @.str.2, i64 6, i1 false) #4
  %1 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.2, i64 6)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

4:                                                ; preds = %0
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(5) getelementptr inbounds nuw (i8, ptr @p, i64 16), ptr noundef nonnull align 1 dereferenceable(5) getelementptr inbounds nuw (i8, ptr @.str.3, i64 1), i64 5, i1 false) #4
  %5 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(5) getelementptr inbounds nuw (i8, ptr @p, i64 16), ptr noundef nonnull dereferenceable(5) @.str.4, i64 5)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %4
  store i8 0, ptr getelementptr inbounds nuw (i8, ptr @p, i64 1), align 1
  %9 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.6, i64 6)
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #5
  unreachable

12:                                               ; preds = %8
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) getelementptr inbounds nuw (i8, ptr @p, i64 3), ptr noundef nonnull align 1 dereferenceable(6) @.str.7, i64 6, i1 false) #4
  %13 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(9) @p, ptr noundef nonnull dereferenceable(9) @.str.8, i64 9)
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %12
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %12
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) getelementptr inbounds nuw (i8, ptr @p, i64 21), ptr noundef nonnull align 1 dereferenceable(3) @.str.9, i64 3, i1 false) #4
  %17 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(5) getelementptr inbounds nuw (i8, ptr @p, i64 19), ptr noundef nonnull dereferenceable(5) @.str.10, i64 5)
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %20, label %19

19:                                               ; preds = %16
  tail call void @abort() #5
  unreachable

20:                                               ; preds = %16
  store <2 x i32> <i32 1145258561, i32 4671045>, ptr @p, align 8
  %21 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(8) @p, ptr noundef nonnull dereferenceable(8) @.str.13, i64 8)
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %24, label %23

23:                                               ; preds = %20
  tail call void @abort() #5
  unreachable

24:                                               ; preds = %20
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(6) @p, ptr noundef nonnull align 1 dereferenceable(6) @.str.2, i64 6, i1 false) #4
  %25 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.2, i64 6)
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %28, label %27

27:                                               ; preds = %24
  tail call void @abort() #5
  unreachable

28:                                               ; preds = %24
  store i32 1, ptr @inside_main, align 4, !tbaa !6
  %29 = load ptr, ptr @s3, align 8, !tbaa !10
  %30 = tail call ptr @strcpy(ptr noundef nonnull dereferenceable(1) getelementptr inbounds nuw (i8, ptr @p, i64 3), ptr noundef nonnull dereferenceable(1) %29)
  %31 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @p, ptr noundef nonnull dereferenceable(6) @.str.14, i64 6)
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %34, label %33

33:                                               ; preds = %28
  tail call void @abort() #5
  unreachable

34:                                               ; preds = %28
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strcpy(ptr noalias returned writeonly, ptr noalias readonly captures(none)) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
