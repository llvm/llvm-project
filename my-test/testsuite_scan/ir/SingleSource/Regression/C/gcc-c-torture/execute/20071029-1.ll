; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071029-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071029-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.T = type { %struct.anon }
%struct.anon = type { i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i32 }

@test.i = internal unnamed_addr global i32 11, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @test(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i32, ptr %0, align 8, !tbaa !6
  %3 = load i32, ptr @test.i, align 4, !tbaa !9
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @test.i, align 4, !tbaa !9
  %5 = icmp eq i32 %2, %3
  br i1 %5, label %7, label %6

6:                                                ; preds = %1
  tail call void @abort() #7
  unreachable

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %9 = load i32, ptr %8, align 4, !tbaa !6
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %47

11:                                               ; preds = %7
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %13 = load i32, ptr %12, align 8, !tbaa !6
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %15, label %47

15:                                               ; preds = %11
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %17 = load i32, ptr %16, align 4, !tbaa !6
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %47

19:                                               ; preds = %15
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %21 = load i32, ptr %20, align 8, !tbaa !6
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %23, label %47

23:                                               ; preds = %19
  %24 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %25 = load i32, ptr %24, align 4, !tbaa !6
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %27, label %47

27:                                               ; preds = %23
  %28 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %29 = load i32, ptr %28, align 8, !tbaa !6
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %47

31:                                               ; preds = %27
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %33 = load i32, ptr %32, align 4, !tbaa !6
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %35, label %47

35:                                               ; preds = %31
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %37 = load i64, ptr %36, align 8, !tbaa !6
  %38 = icmp eq i64 %37, 0
  br i1 %38, label %39, label %47

39:                                               ; preds = %35
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %41 = load i64, ptr %40, align 8, !tbaa !6
  %42 = icmp eq i64 %41, 0
  br i1 %42, label %43, label %47

43:                                               ; preds = %39
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %45 = load i32, ptr %44, align 8, !tbaa !6
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %48, label %47

47:                                               ; preds = %43, %39, %35, %31, %27, %23, %19, %15, %11, %7
  tail call void @abort() #7
  unreachable

48:                                               ; preds = %43
  %49 = icmp eq i32 %4, 20
  br i1 %49, label %50, label %51

50:                                               ; preds = %48
  tail call void @exit(i32 noundef 0) #7
  unreachable

51:                                               ; preds = %48
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

; Function Attrs: nofree noinline noreturn nounwind uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #3 {
  %2 = alloca %union.T, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 4
  br label %4

4:                                                ; preds = %4, %1
  %5 = phi i32 [ %0, %1 ], [ %6, %4 ]
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(52) %3, i8 0, i64 52, i1 false)
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr %2, align 8
  call void @test(ptr noundef nonnull %2)
  br label %4
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 {
  tail call void @foo(i32 noundef 10)
  unreachable
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { noreturn nounwind }
attributes #8 = { nounwind }

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
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
