; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030313-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030313-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { i64, i64, i64, i64 }

@x = dso_local local_unnamed_addr global %struct.A { i64 13, i64 14, i64 15, i64 16 }, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(ptr noundef readonly captures(none) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %1, 12
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @abort() #5
  unreachable

5:                                                ; preds = %2
  %6 = load i64, ptr %0, align 8, !tbaa !6
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %12

8:                                                ; preds = %5
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load i64, ptr %9, align 8, !tbaa !6
  %11 = icmp eq i64 %10, 11
  br i1 %11, label %13, label %12

12:                                               ; preds = %8, %5
  tail call void @abort() #5
  unreachable

13:                                               ; preds = %8
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %15 = load i64, ptr %14, align 8, !tbaa !6
  %16 = icmp eq i64 %15, 2
  br i1 %16, label %17, label %21

17:                                               ; preds = %13
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %19 = load i64, ptr %18, align 8, !tbaa !6
  %20 = icmp eq i64 %19, 12
  br i1 %20, label %22, label %21

21:                                               ; preds = %17, %13
  tail call void @abort() #5
  unreachable

22:                                               ; preds = %17
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %24 = load i64, ptr %23, align 8, !tbaa !6
  %25 = icmp eq i64 %24, 3
  br i1 %25, label %26, label %30

26:                                               ; preds = %22
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %28 = load i64, ptr %27, align 8, !tbaa !6
  %29 = icmp eq i64 %28, 13
  br i1 %29, label %31, label %30

30:                                               ; preds = %26, %22
  tail call void @abort() #5
  unreachable

31:                                               ; preds = %26
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %33 = load i64, ptr %32, align 8, !tbaa !6
  %34 = icmp eq i64 %33, 4
  br i1 %34, label %35, label %39

35:                                               ; preds = %31
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %37 = load i64, ptr %36, align 8, !tbaa !6
  %38 = icmp eq i64 %37, 14
  br i1 %38, label %40, label %39

39:                                               ; preds = %35, %31
  tail call void @abort() #5
  unreachable

40:                                               ; preds = %35
  %41 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %42 = load i64, ptr %41, align 8, !tbaa !6
  %43 = icmp eq i64 %42, 5
  br i1 %43, label %44, label %48

44:                                               ; preds = %40
  %45 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %46 = load i64, ptr %45, align 8, !tbaa !6
  %47 = icmp eq i64 %46, 15
  br i1 %47, label %49, label %48

48:                                               ; preds = %44, %40
  tail call void @abort() #5
  unreachable

49:                                               ; preds = %44
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %51 = load i64, ptr %50, align 8, !tbaa !6
  %52 = icmp eq i64 %51, 6
  br i1 %52, label %53, label %57

53:                                               ; preds = %49
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 88
  %55 = load i64, ptr %54, align 8, !tbaa !6
  %56 = icmp eq i64 %55, 16
  br i1 %56, label %58, label %57

57:                                               ; preds = %53, %49
  tail call void @abort() #5
  unreachable

58:                                               ; preds = %53
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [40 x i64], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  store i64 1, ptr %1, align 8, !tbaa !6
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i64 11, ptr %2, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i64 2, ptr %3, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 12, ptr %4, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store i64 3, ptr %5, align 8, !tbaa !6
  %6 = load i64, ptr @x, align 8, !tbaa !10
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 40
  store i64 %6, ptr %7, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store i64 4, ptr %8, align 8, !tbaa !6
  %9 = load i64, ptr getelementptr inbounds nuw (i8, ptr @x, i64 8), align 8, !tbaa !12
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 56
  store i64 %9, ptr %10, align 8, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 64
  store i64 5, ptr %11, align 8, !tbaa !6
  %12 = load i64, ptr getelementptr inbounds nuw (i8, ptr @x, i64 16), align 8, !tbaa !13
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 72
  store i64 %12, ptr %13, align 8, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 80
  store i64 6, ptr %14, align 8, !tbaa !6
  %15 = load i64, ptr getelementptr inbounds nuw (i8, ptr @x, i64 24), align 8, !tbaa !14
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 88
  store i64 %15, ptr %16, align 8, !tbaa !6
  call void @foo(ptr noundef nonnull %1, i32 noundef 12)
  tail call void @exit(i32 noundef 0) #5
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !7, i64 0}
!11 = !{!"A", !7, i64 0, !7, i64 8, !7, i64 16, !7, i64 24}
!12 = !{!11, !7, i64 8}
!13 = !{!11, !7, i64 16}
!14 = !{!11, !7, i64 24}
