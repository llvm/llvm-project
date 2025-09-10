; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncpy.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncpy.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [12 x i8] c"hello world\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"hellXXX\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c" worXXX\00", align 1
@.str.3 = private unnamed_addr constant [4 x i8] c"XXX\00", align 1
@.str.4 = private unnamed_addr constant [16 x i8] c"hello world\00XXX\00", align 1
@i = dso_local local_unnamed_addr global i32 0, align 4
@.str.7 = private unnamed_addr constant [8 x i8] c"bar\00XXX\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = alloca [64 x i8], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(60) %2, i8 88, i64 60, i1 false)
  store i32 1819043176, ptr %1, align 4
  %3 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %1, ptr noundef nonnull dereferenceable(7) @.str.1, i64 7)
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

6:                                                ; preds = %0
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(64) %1, i8 88, i64 64, i1 false)
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 1819043176, ptr %7, align 4
  %8 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %7, ptr noundef nonnull dereferenceable(7) @.str.1, i64 7)
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @abort() #7
  unreachable

11:                                               ; preds = %6
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(64) %1, i8 88, i64 64, i1 false)
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store i32 1919907616, ptr %12, align 4
  %13 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %12, ptr noundef nonnull dereferenceable(7) @.str.2, i64 7)
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %11
  tail call void @abort() #7
  unreachable

16:                                               ; preds = %11
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(64) %1, i8 88, i64 64, i1 false)
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 1
  store i32 1919907616, ptr %17, align 1
  %18 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %17, ptr noundef nonnull dereferenceable(7) @.str.2, i64 7)
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %21, label %20

20:                                               ; preds = %16
  tail call void @abort() #7
  unreachable

21:                                               ; preds = %16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(64) %1, i8 88, i64 64, i1 false)
  %22 = call i32 @bcmp(ptr noundef nonnull dereferenceable(3) %1, ptr noundef nonnull dereferenceable(3) @.str.3, i64 3)
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %25, label %24

24:                                               ; preds = %21
  tail call void @abort() #7
  unreachable

25:                                               ; preds = %21
  %26 = call i32 @bcmp(ptr noundef nonnull dereferenceable(3) %17, ptr noundef nonnull dereferenceable(3) @.str.3, i64 3)
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %29, label %28

28:                                               ; preds = %25
  tail call void @abort() #7
  unreachable

29:                                               ; preds = %25
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 6
  %31 = call i32 @bcmp(ptr noundef nonnull dereferenceable(3) %30, ptr noundef nonnull dereferenceable(3) @.str.3, i64 3)
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %34, label %33

33:                                               ; preds = %29
  tail call void @abort() #7
  unreachable

34:                                               ; preds = %29
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(52) %35, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 noundef 12, i1 false) #6
  %36 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.4, i64 15)
  %37 = icmp eq i32 %36, 0
  br i1 %37, label %39, label %38

38:                                               ; preds = %34
  tail call void @abort() #7
  unreachable

39:                                               ; preds = %34
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(60) %40, i8 88, i64 60, i1 false)
  store i32 1819043176, ptr %1, align 4
  %41 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %1, ptr noundef nonnull dereferenceable(7) @.str.1, i64 7)
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %44, label %43

43:                                               ; preds = %39
  tail call void @abort() #7
  unreachable

44:                                               ; preds = %39
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(60) %45, i8 88, i64 60, i1 false)
  %46 = load i32, ptr @i, align 4, !tbaa !6
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @i, align 4, !tbaa !6
  %48 = icmp eq i32 %46, 0
  %49 = select i1 %48, i32 7496034, i32 7303014
  store i32 %49, ptr %1, align 4
  %50 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %1, ptr noundef nonnull dereferenceable(7) @.str.7, i64 7)
  %51 = icmp ne i32 %50, 0
  %52 = icmp ne i32 %46, 0
  %53 = select i1 %51, i1 true, i1 %52
  br i1 %53, label %54, label %55

54:                                               ; preds = %44
  tail call void @abort() #7
  unreachable

55:                                               ; preds = %44
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

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
