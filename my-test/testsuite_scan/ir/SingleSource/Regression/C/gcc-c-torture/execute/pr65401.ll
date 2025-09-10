; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65401.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65401.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { [64 x i16] }

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @foo(ptr noundef captures(none) %0) local_unnamed_addr #0 {
  br label %2

2:                                                ; preds = %1, %2
  %3 = phi i64 [ 0, %1 ], [ %14, %2 ]
  %4 = shl nuw nsw i64 %3, 1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 %4
  %6 = load i8, ptr %5, align 1, !tbaa !6
  %7 = zext i8 %6 to i16
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 1
  %9 = load i8, ptr %8, align 1, !tbaa !6
  %10 = zext i8 %9 to i16
  %11 = shl nuw i16 %10, 8
  %12 = or disjoint i16 %11, %7
  %13 = getelementptr inbounds nuw i16, ptr %0, i64 %3
  store i16 %12, ptr %13, align 2, !tbaa !9
  %14 = add nuw nsw i64 %3, 1
  %15 = icmp eq i64 %14, 64
  br i1 %15, label %16, label %2, !llvm.loop !11

16:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @bar(ptr noundef captures(none) %0) local_unnamed_addr #0 {
  br label %2

2:                                                ; preds = %1, %2
  %3 = phi i64 [ 0, %1 ], [ %14, %2 ]
  %4 = shl nuw nsw i64 %3, 1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 %4
  %6 = load i8, ptr %5, align 1, !tbaa !6
  %7 = zext i8 %6 to i16
  %8 = shl nuw i16 %7, 8
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 1
  %10 = load i8, ptr %9, align 1, !tbaa !6
  %11 = zext i8 %10 to i16
  %12 = or disjoint i16 %8, %11
  %13 = getelementptr inbounds nuw i16, ptr %0, i64 %3
  store i16 %12, ptr %13, align 2, !tbaa !9
  %14 = add nuw nsw i64 %3, 1
  %15 = icmp eq i64 %14, 64
  br i1 %15, label %16, label %2, !llvm.loop !13

16:                                               ; preds = %2
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %struct.S, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  store <8 x i16> <i16 16384, i16 16129, i16 15874, i16 15619, i16 15364, i16 15109, i16 14854, i16 14599>, ptr %1, align 16, !tbaa !9
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store <8 x i16> <i16 14344, i16 14089, i16 13834, i16 13579, i16 13324, i16 13069, i16 12814, i16 12559>, ptr %2, align 16, !tbaa !9
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store <8 x i16> <i16 12304, i16 12049, i16 11794, i16 11539, i16 11284, i16 11029, i16 10774, i16 10519>, ptr %3, align 16, !tbaa !9
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store <8 x i16> <i16 10264, i16 10009, i16 9754, i16 9499, i16 9244, i16 8989, i16 8734, i16 8479>, ptr %4, align 16, !tbaa !9
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 64
  store <8 x i16> <i16 8224, i16 7969, i16 7714, i16 7459, i16 7204, i16 6949, i16 6694, i16 6439>, ptr %5, align 16, !tbaa !9
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 80
  store <8 x i16> <i16 6184, i16 5929, i16 5674, i16 5419, i16 5164, i16 4909, i16 4654, i16 4399>, ptr %6, align 16, !tbaa !9
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 96
  store <8 x i16> <i16 4144, i16 3889, i16 3634, i16 3379, i16 3124, i16 2869, i16 2614, i16 2359>, ptr %7, align 16, !tbaa !9
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 112
  store <8 x i16> <i16 2104, i16 1849, i16 1594, i16 1339, i16 1084, i16 829, i16 574, i16 319>, ptr %8, align 16, !tbaa !9
  call void @foo(ptr noundef nonnull %1)
  br label %13

9:                                                ; preds = %13
  %10 = add nuw nsw i64 %14, 1
  %11 = icmp eq i64 %10, 64
  br i1 %11, label %12, label %13, !llvm.loop !14

12:                                               ; preds = %9
  store <8 x i16> <i16 16384, i16 16129, i16 15874, i16 15619, i16 15364, i16 15109, i16 14854, i16 14599>, ptr %1, align 16, !tbaa !9
  store <8 x i16> <i16 14344, i16 14089, i16 13834, i16 13579, i16 13324, i16 13069, i16 12814, i16 12559>, ptr %2, align 16, !tbaa !9
  store <8 x i16> <i16 12304, i16 12049, i16 11794, i16 11539, i16 11284, i16 11029, i16 10774, i16 10519>, ptr %3, align 16, !tbaa !9
  store <8 x i16> <i16 10264, i16 10009, i16 9754, i16 9499, i16 9244, i16 8989, i16 8734, i16 8479>, ptr %4, align 16, !tbaa !9
  store <8 x i16> <i16 8224, i16 7969, i16 7714, i16 7459, i16 7204, i16 6949, i16 6694, i16 6439>, ptr %5, align 16, !tbaa !9
  store <8 x i16> <i16 6184, i16 5929, i16 5674, i16 5419, i16 5164, i16 4909, i16 4654, i16 4399>, ptr %6, align 16, !tbaa !9
  store <8 x i16> <i16 4144, i16 3889, i16 3634, i16 3379, i16 3124, i16 2869, i16 2614, i16 2359>, ptr %7, align 16, !tbaa !9
  store <8 x i16> <i16 2104, i16 1849, i16 1594, i16 1339, i16 1084, i16 829, i16 574, i16 319>, ptr %8, align 16, !tbaa !9
  call void @bar(ptr noundef nonnull %1)
  br label %25

13:                                               ; preds = %0, %9
  %14 = phi i64 [ 0, %0 ], [ %10, %9 ]
  %15 = getelementptr inbounds nuw i16, ptr %1, i64 %14
  %16 = load i16, ptr %15, align 2, !tbaa !9
  %17 = mul nsw i64 %14, -255
  %18 = add nsw i64 %17, 16384
  %19 = zext i16 %16 to i64
  %20 = icmp eq i64 %18, %19
  br i1 %20, label %9, label %21

21:                                               ; preds = %13
  tail call void @abort() #5
  unreachable

22:                                               ; preds = %25
  %23 = add nuw nsw i64 %26, 1
  %24 = icmp eq i64 %23, 64
  br i1 %24, label %34, label %25, !llvm.loop !15

25:                                               ; preds = %12, %22
  %26 = phi i64 [ 0, %12 ], [ %23, %22 ]
  %27 = getelementptr inbounds nuw i16, ptr %1, i64 %26
  %28 = load i16, ptr %27, align 2, !tbaa !9
  %29 = mul nuw nsw i64 %26, 255
  %30 = add nuw nsw i64 %29, 64
  %31 = zext i16 %28 to i64
  %32 = icmp eq i64 %30, %31
  br i1 %32, label %22, label %33

33:                                               ; preds = %25
  tail call void @abort() #5
  unreachable

34:                                               ; preds = %22
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"short", !7, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
!14 = distinct !{!14, !12}
!15 = distinct !{!15, !12}
