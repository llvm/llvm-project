; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20090113-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20090113-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.bitmap_element_def = type { ptr, ptr, i32, [2 x i64] }
%struct.bitmap_iterator = type { ptr, ptr, i32, i64 }

@bitmap_zero_bits = dso_local global %struct.bitmap_element_def zeroinitializer, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca %struct.bitmap_element_def, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(40) %1, i8 0, i64 24, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store <2 x i64> splat (i64 1), ptr %2, align 8
  call fastcc void @foobar(ptr nonnull %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #8
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: nofree noinline nounwind uwtable
define internal fastcc void @foobar(ptr %0) unnamed_addr #3 {
  %2 = alloca %struct.bitmap_iterator, align 8
  %3 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call fastcc void @bmp_iter_set_init(ptr noundef %2, ptr %0, ptr noundef %3)
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 16
  br label %6

6:                                                ; preds = %55, %1
  %7 = load i32, ptr %3, align 4, !tbaa !6
  %8 = load i64, ptr %4, align 8, !tbaa !10
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %22, label %10

10:                                               ; preds = %6
  %11 = and i64 %8, 1
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %13, label %20

13:                                               ; preds = %10, %13
  %14 = phi i64 [ %16, %13 ], [ %8, %10 ]
  %15 = phi i32 [ %17, %13 ], [ %7, %10 ]
  %16 = lshr exact i64 %14, 1
  %17 = add i32 %15, 1
  %18 = and i64 %14, 2
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %13, label %20, !llvm.loop !15

20:                                               ; preds = %13, %10
  %21 = phi i32 [ %7, %10 ], [ %17, %13 ]
  store i32 %21, ptr %3, align 4, !tbaa !6
  br label %55

22:                                               ; preds = %6
  %23 = add i32 %7, 63
  %24 = and i32 %23, -64
  %25 = load i32, ptr %5, align 8, !tbaa !17
  %26 = add i32 %25, 1
  %27 = load ptr, ptr %2, align 8, !tbaa !18
  br label %28

28:                                               ; preds = %51, %22
  %29 = phi i32 [ %24, %22 ], [ %54, %51 ]
  %30 = phi i32 [ %26, %22 ], [ 0, %51 ]
  %31 = phi ptr [ %27, %22 ], [ %49, %51 ]
  %32 = icmp eq i32 %30, 2
  br i1 %32, label %48, label %33

33:                                               ; preds = %28
  %34 = getelementptr inbounds nuw i8, ptr %31, i64 24
  br label %35

35:                                               ; preds = %44, %33
  %36 = phi i32 [ %29, %33 ], [ %45, %44 ]
  %37 = phi i32 [ %30, %33 ], [ %46, %44 ]
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds nuw i64, ptr %34, i64 %38
  %40 = load i64, ptr %39, align 8, !tbaa !19
  %41 = icmp eq i64 %40, 0
  br i1 %41, label %44, label %42

42:                                               ; preds = %35
  store i64 %40, ptr %4, align 8, !tbaa !10
  store i32 %36, ptr %3, align 4
  store i32 %37, ptr %5, align 8
  store ptr %31, ptr %2, align 8, !tbaa !18
  call fastcc void @bmp_iter_set_tail(ptr noundef nonnull %2, ptr noundef nonnull %3)
  %43 = load i32, ptr %3, align 4, !tbaa !6
  br label %55

44:                                               ; preds = %35
  %45 = add i32 %36, 64
  %46 = add i32 %37, 1
  %47 = icmp eq i32 %46, 2
  br i1 %47, label %48, label %35, !llvm.loop !20

48:                                               ; preds = %44, %28
  %49 = load ptr, ptr %31, align 8, !tbaa !21
  %50 = icmp eq ptr %49, null
  br i1 %50, label %57, label %51

51:                                               ; preds = %48
  %52 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %53 = load i32, ptr %52, align 8, !tbaa !23
  %54 = shl i32 %53, 7
  br label %28

55:                                               ; preds = %20, %42
  %56 = phi i32 [ %21, %20 ], [ %43, %42 ]
  tail call fastcc void @catchme(i32 noundef %56)
  call fastcc void @bmp_iter_next(ptr noundef %2, ptr noundef %3)
  br label %6, !llvm.loop !24

57:                                               ; preds = %48
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define internal fastcc void @bmp_iter_set_init(ptr noundef nonnull writeonly captures(none) initializes((0, 20), (24, 32)) %0, ptr %1, ptr noundef nonnull writeonly captures(none) initializes((0, 4)) %2) unnamed_addr #4 {
  store ptr %1, ptr %0, align 8, !tbaa !18
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr null, ptr %4, align 8, !tbaa !25
  %5 = icmp eq ptr %1, null
  br i1 %5, label %6, label %7

6:                                                ; preds = %3
  store ptr @bitmap_zero_bits, ptr %0, align 8, !tbaa !18
  br label %7

7:                                                ; preds = %3, %6
  %8 = phi ptr [ %1, %3 ], [ @bitmap_zero_bits, %6 ]
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %10 = load i32, ptr %9, align 8, !tbaa !23
  %11 = shl i32 %10, 7
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i32 0, ptr %12, align 8, !tbaa !17
  %13 = getelementptr inbounds nuw i8, ptr %8, i64 24
  %14 = load i64, ptr %13, align 8, !tbaa !19
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store i64 %14, ptr %15, align 8, !tbaa !10
  %16 = icmp eq i64 %14, 0
  %17 = zext i1 %16 to i32
  %18 = or disjoint i32 %11, %17
  store i32 %18, ptr %2, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define internal fastcc void @catchme(i32 noundef %0) unnamed_addr #3 {
  %2 = and i32 %0, -65
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @abort() #9
  unreachable

5:                                                ; preds = %1
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define internal fastcc void @bmp_iter_next(ptr noundef nonnull captures(none) %0, ptr noundef nonnull captures(none) %1) unnamed_addr #5 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %4 = load i64, ptr %3, align 8, !tbaa !10
  %5 = lshr i64 %4, 1
  store i64 %5, ptr %3, align 8, !tbaa !10
  %6 = load i32, ptr %1, align 4, !tbaa !6
  %7 = add i32 %6, 1
  store i32 %7, ptr %1, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define internal fastcc void @bmp_iter_set_tail(ptr noundef nonnull captures(none) %0, ptr noundef nonnull captures(none) %1) unnamed_addr #6 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %4 = load i64, ptr %3, align 8, !tbaa !10
  %5 = and i64 %4, 1
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %7, label %17

7:                                                ; preds = %2
  %8 = load i32, ptr %1, align 4, !tbaa !6
  br label %9

9:                                                ; preds = %7, %9
  %10 = phi i32 [ %8, %7 ], [ %13, %9 ]
  %11 = phi i64 [ %4, %7 ], [ %12, %9 ]
  %12 = lshr exact i64 %11, 1
  %13 = add i32 %10, 1
  %14 = and i64 %11, 2
  %15 = icmp eq i64 %14, 0
  br i1 %15, label %9, label %16, !llvm.loop !26

16:                                               ; preds = %9
  store i64 %12, ptr %3, align 8, !tbaa !10
  store i32 %13, ptr %1, align 4, !tbaa !6
  br label %17

17:                                               ; preds = %16, %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #7

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind }
attributes #9 = { noreturn nounwind }

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
!10 = !{!11, !14, i64 24}
!11 = !{!"", !12, i64 0, !12, i64 8, !7, i64 16, !14, i64 24}
!12 = !{!"p1 _ZTS18bitmap_element_def", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
!14 = !{!"long", !8, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!11, !7, i64 16}
!18 = !{!11, !12, i64 0}
!19 = !{!14, !14, i64 0}
!20 = distinct !{!20, !16}
!21 = !{!22, !12, i64 0}
!22 = !{!"bitmap_element_def", !12, i64 0, !12, i64 8, !7, i64 16, !8, i64 24}
!23 = !{!22, !7, i64 16}
!24 = distinct !{!24, !16}
!25 = !{!11, !12, i64 8}
!26 = distinct !{!26, !16}
