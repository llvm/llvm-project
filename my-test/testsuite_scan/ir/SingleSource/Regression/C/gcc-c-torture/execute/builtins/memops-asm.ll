; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memops-asm.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memops-asm.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { [32 x i8] }

@a = dso_local local_unnamed_addr global %struct.A { [32 x i8] c"foobar\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00" }, align 1
@x = dso_local global [64 x i8] c"foobar\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
@i = dso_local local_unnamed_addr global i32 39, align 4
@j = dso_local local_unnamed_addr global i32 6, align 4
@k = dso_local local_unnamed_addr global i32 4, align 4
@__const.main_test.c = private unnamed_addr constant { <{ i8, [31 x i8] }> } { <{ i8, [31 x i8] }> <{ i8 120, [31 x i8] zeroinitializer }> }, align 1
@inside_main = external local_unnamed_addr global i32, align 4
@y = dso_local global [64 x i8] zeroinitializer, align 1
@.str = private unnamed_addr constant [13 x i8] c"foXXXXfoobar\00", align 1
@.str.1 = private unnamed_addr constant [13 x i8] c"fooXXXXfobar\00", align 1
@.str.2 = private unnamed_addr constant [13 x i8] c"fooX\00\00Xfobar\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = alloca %struct.A, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(32) %1, ptr noundef nonnull align 1 dereferenceable(32) @a, i64 32, i1 false), !tbaa.struct !6
  store i32 1, ptr @inside_main, align 4, !tbaa !10
  %2 = call i32 @bcmp(ptr noundef nonnull dereferenceable(32) %1, ptr noundef nonnull dereferenceable(32) @x, i64 32)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %7

4:                                                ; preds = %0
  %5 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(31) getelementptr inbounds nuw (i8, ptr @__const.main_test.c, i64 1), ptr noundef nonnull dereferenceable(31) getelementptr inbounds nuw (i8, ptr @x, i64 32), i64 31)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %8, label %7

7:                                                ; preds = %4, %0
  tail call void @abort() #8
  unreachable

8:                                                ; preds = %4
  %9 = load i32, ptr @i, align 4, !tbaa !10
  %10 = sext i32 %9 to i64
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 @y, ptr nonnull align 1 @x, i64 %10, i1 false)
  %11 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(64) @x, ptr noundef nonnull dereferenceable(64) @y, i64 64)
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %14, label %13

13:                                               ; preds = %8
  tail call void @abort() #8
  unreachable

14:                                               ; preds = %8
  %15 = load i32, ptr @j, align 4, !tbaa !10
  %16 = sext i32 %15 to i64
  %17 = tail call ptr @my_memcpy(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @y, i64 6), ptr noundef nonnull @x, i64 noundef %16) #7
  %18 = icmp eq ptr %17, getelementptr inbounds nuw (i8, ptr @y, i64 6)
  br i1 %18, label %19, label %25

19:                                               ; preds = %14
  %20 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(6) @x, ptr noundef nonnull dereferenceable(6) @y, i64 6)
  %21 = icmp eq i32 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %19
  %23 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(58) @x, ptr noundef nonnull dereferenceable(58) getelementptr inbounds nuw (i8, ptr @y, i64 6), i64 58)
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %26, label %25

25:                                               ; preds = %22, %19, %14
  tail call void @abort() #8
  unreachable

26:                                               ; preds = %22
  %27 = load i32, ptr @k, align 4, !tbaa !10
  %28 = sext i32 %27 to i64
  tail call void @llvm.memset.p0.i64(ptr nonnull align 1 getelementptr inbounds nuw (i8, ptr @y, i64 2), i8 88, i64 %28, i1 false)
  %29 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(13) @y, ptr noundef nonnull dereferenceable(13) @.str, i64 13)
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %32, label %31

31:                                               ; preds = %26
  tail call void @abort() #8
  unreachable

32:                                               ; preds = %26
  tail call void @my_bcopy(ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @y, i64 1), ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @y, i64 2), i64 noundef 6) #7
  %33 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(13) @y, ptr noundef nonnull dereferenceable(13) @.str.1, i64 13)
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %36, label %35

35:                                               ; preds = %32
  tail call void @abort() #8
  unreachable

36:                                               ; preds = %32
  store i16 0, ptr getelementptr inbounds nuw (i8, ptr @y, i64 4), align 1
  %37 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(13) @y, ptr noundef nonnull dereferenceable(13) @.str.2, i64 13)
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %36
  tail call void @abort() #8
  unreachable

40:                                               ; preds = %36
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nounwind
declare ptr @my_memcpy(ptr noundef, ptr noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

; Function Attrs: nounwind
declare void @my_bcopy(ptr noundef, ptr noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #6

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 0, i64 32, !7}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
