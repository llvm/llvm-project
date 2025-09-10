; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20000706-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20000706-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.baz = type { i32, i32, i32, i32, i32 }

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(ptr noundef readonly captures(none) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = load i32, ptr %0, align 4, !tbaa !6
  %8 = icmp eq i32 %7, 1
  br i1 %8, label %9, label %35

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %11 = load i32, ptr %10, align 4, !tbaa !11
  %12 = icmp eq i32 %11, 2
  br i1 %12, label %13, label %35

13:                                               ; preds = %9
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %15 = load i32, ptr %14, align 4, !tbaa !12
  %16 = icmp eq i32 %15, 3
  br i1 %16, label %17, label %35

17:                                               ; preds = %13
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %19 = load i32, ptr %18, align 4, !tbaa !13
  %20 = icmp eq i32 %19, 4
  br i1 %20, label %21, label %35

21:                                               ; preds = %17
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %23 = load i32, ptr %22, align 4, !tbaa !14
  %24 = icmp ne i32 %23, 5
  %25 = icmp ne i32 %1, 6
  %26 = or i1 %25, %24
  %27 = icmp ne i32 %2, 7
  %28 = or i1 %27, %26
  %29 = icmp ne i32 %3, 8
  %30 = or i1 %29, %28
  %31 = icmp ne i32 %4, 9
  %32 = or i1 %31, %30
  %33 = icmp ne i32 %5, 10
  %34 = or i1 %33, %32
  br i1 %34, label %35, label %36

35:                                               ; preds = %21, %17, %13, %9, %6
  tail call void @abort() #5
  unreachable

36:                                               ; preds = %21
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(ptr dead_on_return noundef readonly captures(none) %0, ptr readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr %0, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %5, label %21

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %7 = load i32, ptr %6, align 4, !tbaa !11
  %8 = icmp eq i32 %7, 2
  br i1 %8, label %9, label %21

9:                                                ; preds = %5
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = load i32, ptr %10, align 4, !tbaa !12
  %12 = icmp eq i32 %11, 3
  br i1 %12, label %13, label %21

13:                                               ; preds = %9
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %15 = load i32, ptr %14, align 4, !tbaa !13
  %16 = icmp eq i32 %15, 4
  br i1 %16, label %17, label %21

17:                                               ; preds = %13
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %19 = load i32, ptr %18, align 4, !tbaa !14
  %20 = icmp eq i32 %19, 5
  br i1 %20, label %22, label %21

21:                                               ; preds = %17, %13, %9, %5, %2
  tail call void @abort() #5
  unreachable

22:                                               ; preds = %17
  ret void
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %struct.baz, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  store i32 1, ptr %1, align 4, !tbaa !15
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 2, ptr %2, align 4, !tbaa !15
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 3, ptr %3, align 4, !tbaa !15
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 4, ptr %4, align 4, !tbaa !15
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 5, ptr %5, align 4, !tbaa !15
  call void @foo(ptr dead_on_return noundef nonnull %1, ptr poison)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  tail call void @exit(i32 noundef 0) #5
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

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
!6 = !{!7, !8, i64 0}
!7 = !{!"baz", !8, i64 0, !8, i64 4, !8, i64 8, !8, i64 12, !8, i64 16}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 4}
!12 = !{!7, !8, i64 8}
!13 = !{!7, !8, i64 12}
!14 = !{!7, !8, i64 16}
!15 = !{!8, !8, i64 0}
