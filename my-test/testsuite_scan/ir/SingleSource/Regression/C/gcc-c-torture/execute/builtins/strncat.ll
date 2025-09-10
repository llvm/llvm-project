; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncat.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncat.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@x = dso_local local_unnamed_addr global i32 123, align 4
@.str = private unnamed_addr constant [12 x i8] c"hello world\00", align 1
@.str.1 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.2 = private unnamed_addr constant [16 x i8] c"hello world\00XXX\00", align 1
@.str.4 = private unnamed_addr constant [19 x i8] c"hello worldfoo\00XXX\00", align 1
@.str.5 = private unnamed_addr constant [27 x i8] c"hello worldhello world\00XXX\00", align 1
@.str.6 = private unnamed_addr constant [22 x i8] c"hello world world\00XXX\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @main_test() local_unnamed_addr #0 {
  %1 = alloca [64 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %2, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %3 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #8
  unreachable

6:                                                ; preds = %0
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %7, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %8 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @abort() #8
  unreachable

11:                                               ; preds = %6
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %12, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %14 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %17, label %16

16:                                               ; preds = %11
  tail call void @abort() #8
  unreachable

17:                                               ; preds = %11
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %18, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 6
  %20 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %21 = icmp eq i32 %20, 0
  br i1 %21, label %23, label %22

22:                                               ; preds = %17
  tail call void @abort() #8
  unreachable

23:                                               ; preds = %17
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %24, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %25 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %28, label %27

27:                                               ; preds = %23
  tail call void @abort() #8
  unreachable

28:                                               ; preds = %23
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %29, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %30 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %33, label %32

32:                                               ; preds = %28
  tail call void @abort() #8
  unreachable

33:                                               ; preds = %28
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %34, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %35 = load i32, ptr @x, align 4, !tbaa !6
  %36 = add nsw i32 %35, 1
  store i32 %36, ptr @x, align 4, !tbaa !6
  %37 = sext i32 %36 to i64
  %38 = call ptr @strncat(ptr noundef nonnull dereferenceable(1) %19, ptr noundef nonnull @.str.1, i64 noundef %37) #7
  %39 = icmp eq i32 %36, 124
  br i1 %39, label %40, label %43

40:                                               ; preds = %33
  %41 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %44, label %43

43:                                               ; preds = %40, %33
  call void @abort() #8
  unreachable

44:                                               ; preds = %40
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %45, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %46 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %47 = getelementptr inbounds i8, ptr %1, i64 %46
  store i32 7303014, ptr %47, align 1
  %48 = call i32 @bcmp(ptr noundef nonnull dereferenceable(18) %1, ptr noundef nonnull dereferenceable(18) @.str.4, i64 18)
  %49 = icmp eq i32 %48, 0
  br i1 %49, label %51, label %50

50:                                               ; preds = %44
  call void @abort() #8
  unreachable

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %52, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %53 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %54 = getelementptr inbounds i8, ptr %1, i64 %53
  store i32 7303014, ptr %54, align 1
  %55 = call i32 @bcmp(ptr noundef nonnull dereferenceable(18) %1, ptr noundef nonnull dereferenceable(18) @.str.4, i64 18)
  %56 = icmp eq i32 %55, 0
  br i1 %56, label %58, label %57

57:                                               ; preds = %51
  call void @abort() #8
  unreachable

58:                                               ; preds = %51
  %59 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %59, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %60 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %1)
  %61 = getelementptr inbounds i8, ptr %1, i64 %60
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %61, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false)
  %62 = call i32 @bcmp(ptr noundef nonnull dereferenceable(26) %1, ptr noundef nonnull dereferenceable(26) @.str.5, i64 26)
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %65, label %64

64:                                               ; preds = %58
  call void @abort() #8
  unreachable

65:                                               ; preds = %58
  %66 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %66, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %67 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %13)
  %68 = getelementptr inbounds i8, ptr %13, i64 %67
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %68, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false)
  %69 = call i32 @bcmp(ptr noundef nonnull dereferenceable(26) %1, ptr noundef nonnull dereferenceable(26) @.str.5, i64 26)
  %70 = icmp eq i32 %69, 0
  br i1 %70, label %72, label %71

71:                                               ; preds = %65
  call void @abort() #8
  unreachable

72:                                               ; preds = %65
  %73 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %73, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %74 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %19)
  %75 = getelementptr inbounds i8, ptr %19, i64 %74
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %75, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false)
  %76 = call i32 @bcmp(ptr noundef nonnull dereferenceable(26) %1, ptr noundef nonnull dereferenceable(26) @.str.5, i64 26)
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %79, label %78

78:                                               ; preds = %72
  call void @abort() #8
  unreachable

79:                                               ; preds = %72
  %80 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %80, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %81 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %19)
  %82 = getelementptr inbounds i8, ptr %19, i64 %81
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(7) %82, ptr noundef nonnull align 1 dereferenceable(7) getelementptr inbounds nuw (i8, ptr @.str, i64 5), i64 7, i1 false)
  %83 = call i32 @bcmp(ptr noundef nonnull dereferenceable(21) %1, ptr noundef nonnull dereferenceable(21) @.str.6, i64 21)
  %84 = icmp eq i32 %83, 0
  br i1 %84, label %86, label %85

85:                                               ; preds = %79
  call void @abort() #8
  unreachable

86:                                               ; preds = %79
  %87 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(52) %87, i8 88, i64 52, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(12) %1, ptr noundef nonnull align 1 dereferenceable(12) @.str, i64 12, i1 false) #7
  %88 = call i32 @bcmp(ptr noundef nonnull dereferenceable(15) %1, ptr noundef nonnull dereferenceable(15) @.str.2, i64 15)
  %89 = icmp eq i32 %88, 0
  br i1 %89, label %91, label %90

90:                                               ; preds = %86
  call void @abort() #8
  unreachable

91:                                               ; preds = %86
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strncat(ptr noalias noundef returned, ptr noalias noundef readonly captures(none), i64 noundef) local_unnamed_addr #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #6

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr captures(none)) local_unnamed_addr #6

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
