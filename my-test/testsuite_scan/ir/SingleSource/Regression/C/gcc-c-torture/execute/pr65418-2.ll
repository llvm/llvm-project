; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65418-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65418-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  switch i32 %0, label %2 [
    i32 -205, label %3
    i32 -211, label %3
    i32 -216, label %3
    i32 -218, label %3
    i32 -223, label %3
  ]

2:                                                ; preds = %1
  br label %3

3:                                                ; preds = %1, %1, %1, %1, %1, %2
  %4 = phi i32 [ 0, %2 ], [ 1, %1 ], [ 1, %1 ], [ 1, %1 ], [ 1, %1 ], [ 1, %1 ]
  ret i32 %4
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store volatile i32 -230, ptr %1, align 4, !tbaa !6
  %2 = load volatile i32, ptr %1, align 4, !tbaa !6
  %3 = icmp slt i32 %2, -200
  br i1 %3, label %4, label %31

4:                                                ; preds = %0, %26
  %5 = load volatile i32, ptr %1, align 4, !tbaa !6
  %6 = tail call i32 @foo(i32 noundef %5)
  %7 = load volatile i32, ptr %1, align 4, !tbaa !6
  %8 = icmp eq i32 %7, -216
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = load volatile i32, ptr %1, align 4, !tbaa !6
  %11 = icmp eq i32 %10, -211
  br i1 %11, label %22, label %12

12:                                               ; preds = %9
  %13 = load volatile i32, ptr %1, align 4, !tbaa !6
  %14 = icmp eq i32 %13, -218
  br i1 %14, label %22, label %15

15:                                               ; preds = %12
  %16 = load volatile i32, ptr %1, align 4, !tbaa !6
  %17 = icmp eq i32 %16, -205
  br i1 %17, label %22, label %18

18:                                               ; preds = %15
  %19 = load volatile i32, ptr %1, align 4, !tbaa !6
  %20 = icmp eq i32 %19, -223
  %21 = zext i1 %20 to i32
  br label %22

22:                                               ; preds = %18, %15, %12, %9, %4
  %23 = phi i32 [ 1, %15 ], [ 1, %12 ], [ 1, %9 ], [ 1, %4 ], [ %21, %18 ]
  %24 = icmp eq i32 %6, %23
  br i1 %24, label %26, label %25

25:                                               ; preds = %22
  tail call void @abort() #4
  unreachable

26:                                               ; preds = %22
  %27 = load volatile i32, ptr %1, align 4, !tbaa !6
  %28 = add nsw i32 %27, 1
  store volatile i32 %28, ptr %1, align 4, !tbaa !6
  %29 = load volatile i32, ptr %1, align 4, !tbaa !6
  %30 = icmp slt i32 %29, -200
  br i1 %30, label %4, label %31, !llvm.loop !10

31:                                               ; preds = %26, %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
