; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcat.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strcat.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [12 x i8] c"hello world\00", align 1
@.str.2 = private unnamed_addr constant [16 x i8] c"hello world\00XXX\00", align 1
@inside_main = external local_unnamed_addr global i32, align 4
@.str.3 = private unnamed_addr constant [6 x i8] c" 1111\00", align 1
@.str.4 = private unnamed_addr constant [21 x i8] c"hello world 1111\00XXX\00", align 1
@.str.5 = private unnamed_addr constant [6 x i8] c" 2222\00", align 1
@.str.6 = private unnamed_addr constant [21 x i8] c"hello world 2222\00XXX\00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c" 3333\00", align 1
@.str.8 = private unnamed_addr constant [21 x i8] c"hello world 3333\00XXX\00", align 1
@.str.11 = private unnamed_addr constant [3 x i8] c"a \00", align 1
@.str.12 = private unnamed_addr constant [5 x i8] c"test\00", align 1
@.str.14 = private unnamed_addr constant [31 x i8] c"hello world: this is a test.\00X\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = alloca [64 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %2, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %3 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

6:                                                ; preds = %0
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %7, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %8 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @abort() #7
  unreachable

11:                                               ; preds = %6
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %12, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %13 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %11
  tail call void @abort() #7
  unreachable

16:                                               ; preds = %11
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %17, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 6
  %19 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %22, label %21

21:                                               ; preds = %16
  tail call void @abort() #7
  unreachable

22:                                               ; preds = %16
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %23, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %24 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %27, label %26

26:                                               ; preds = %22
  tail call void @abort() #7
  unreachable

27:                                               ; preds = %22
  store i32 0, ptr @inside_main, align 4, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %28, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %29 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %30 = getelementptr inbounds i8, ptr %1, i64 %29
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %30, ptr noundef nonnull align 1 dereferenceable(6) @.str.3, i64 6, i1 false)
  %31 = call i32 @bcmp(ptr noundef nonnull dereferenceable(20) %1, ptr noundef nonnull dereferenceable(20) @.str.4, i64 20)
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %34, label %33

33:                                               ; preds = %27
  tail call void @abort() #7
  unreachable

34:                                               ; preds = %27
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %35, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 5
  %37 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %36)
  %38 = getelementptr inbounds i8, ptr %36, i64 %37
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %38, ptr noundef nonnull align 1 dereferenceable(6) @.str.5, i64 6, i1 false)
  %39 = call i32 @bcmp(ptr noundef nonnull dereferenceable(20) %1, ptr noundef nonnull dereferenceable(20) @.str.6, i64 20)
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %42, label %41

41:                                               ; preds = %34
  tail call void @abort() #7
  unreachable

42:                                               ; preds = %34
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %43, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %44 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %18)
  %45 = getelementptr inbounds i8, ptr %18, i64 %44
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %45, ptr noundef nonnull align 1 dereferenceable(6) @.str.7, i64 6, i1 false)
  %46 = call i32 @bcmp(ptr noundef nonnull dereferenceable(20) %1, ptr noundef nonnull dereferenceable(20) @.str.8, i64 20)
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %49, label %48

48:                                               ; preds = %42
  tail call void @abort() #7
  unreachable

49:                                               ; preds = %42
  %50 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %50, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %51 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %52 = getelementptr inbounds i8, ptr %1, i64 %51
  store i64 9134095815942202, ptr %52, align 1
  %53 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %54 = getelementptr inbounds i8, ptr %1, i64 %53
  store i32 2126697, ptr %54, align 1
  %55 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %56 = getelementptr inbounds i8, ptr %1, i64 %55
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(3) %56, ptr noundef nonnull align 1 dereferenceable(3) @.str.11, i64 3, i1 false)
  %57 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %58 = getelementptr inbounds i8, ptr %1, i64 %57
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %58, ptr noundef nonnull align 1 dereferenceable(5) @.str.12, i64 5, i1 false)
  %59 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %60 = getelementptr inbounds i8, ptr %1, i64 %59
  store i16 46, ptr %60, align 1
  %61 = call i32 @bcmp(ptr noundef nonnull dereferenceable(30) %1, ptr noundef nonnull dereferenceable(30) @.str.14, i64 30)
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %64, label %63

63:                                               ; preds = %49
  tail call void @abort() #7
  unreachable

64:                                               ; preds = %49
  store i32 1, ptr @inside_main, align 4, !tbaa !6
  %65 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %65, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #6
  %66 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %67 = icmp eq i32 %66, 0
  br i1 %67, label %69, label %68

68:                                               ; preds = %64
  tail call void @abort() #7
  unreachable

69:                                               ; preds = %64
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr captures(none)) local_unnamed_addr #5

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
