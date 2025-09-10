; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2006-12-04-DynAllocAndRestore.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2006-12-04-DynAllocAndRestore.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%class.BabyDebugTest = type { %class.MamaDebugTest }
%class.MamaDebugTest = type { i32 }

$_ZN13BabyDebugTest4doitEv = comdat any

@_ZN13BabyDebugTest3dohE = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = alloca %class.BabyDebugTest, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #4
  store i32 20, ptr %3, align 4, !tbaa !6
  %4 = call noundef i32 @_ZN13BabyDebugTest4doitEv(ptr noundef nonnull align 4 dereferenceable(4) %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_ZN13BabyDebugTest4doitEv(ptr noundef nonnull align 4 dereferenceable(4) %0) local_unnamed_addr #2 comdat {
  %2 = load i32, ptr %0, align 4, !tbaa !6
  %3 = zext i32 %2 to i64
  %4 = alloca i32, i64 %3, align 4
  %5 = icmp sgt i32 %2, 0
  br i1 %5, label %6, label %52

6:                                                ; preds = %1
  %7 = icmp ult i32 %2, 8
  br i1 %7, label %21, label %8

8:                                                ; preds = %6
  %9 = and i64 %3, 2147483640
  br label %10

10:                                               ; preds = %10, %8
  %11 = phi i64 [ 0, %8 ], [ %16, %10 ]
  %12 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %8 ], [ %17, %10 ]
  %13 = add <4 x i32> %12, splat (i32 4)
  %14 = getelementptr inbounds nuw i32, ptr %4, i64 %11
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 16
  store <4 x i32> %12, ptr %14, align 4, !tbaa !11
  store <4 x i32> %13, ptr %15, align 4, !tbaa !11
  %16 = add nuw i64 %11, 8
  %17 = add <4 x i32> %12, splat (i32 8)
  %18 = icmp eq i64 %16, %9
  br i1 %18, label %19, label %10, !llvm.loop !12

19:                                               ; preds = %10
  %20 = icmp eq i64 %9, %3
  br i1 %20, label %29, label %21

21:                                               ; preds = %6, %19
  %22 = phi i64 [ 0, %6 ], [ %9, %19 ]
  br label %23

23:                                               ; preds = %21, %23
  %24 = phi i64 [ %27, %23 ], [ %22, %21 ]
  %25 = getelementptr inbounds nuw i32, ptr %4, i64 %24
  %26 = trunc nuw nsw i64 %24 to i32
  store i32 %26, ptr %25, align 4, !tbaa !11
  %27 = add nuw nsw i64 %24, 1
  %28 = icmp eq i64 %27, %3
  br i1 %28, label %29, label %23, !llvm.loop !16

29:                                               ; preds = %23, %19
  %30 = icmp ult i32 %2, 8
  br i1 %30, label %49, label %31

31:                                               ; preds = %29
  %32 = and i64 %3, 2147483640
  br label %33

33:                                               ; preds = %33, %31
  %34 = phi i64 [ 0, %31 ], [ %43, %33 ]
  %35 = phi <4 x i32> [ zeroinitializer, %31 ], [ %41, %33 ]
  %36 = phi <4 x i32> [ zeroinitializer, %31 ], [ %42, %33 ]
  %37 = getelementptr inbounds nuw i32, ptr %4, i64 %34
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 16
  %39 = load <4 x i32>, ptr %37, align 4, !tbaa !11
  %40 = load <4 x i32>, ptr %38, align 4, !tbaa !11
  %41 = add <4 x i32> %39, %35
  %42 = add <4 x i32> %40, %36
  %43 = add nuw i64 %34, 8
  %44 = icmp eq i64 %43, %32
  br i1 %44, label %45, label %33, !llvm.loop !17

45:                                               ; preds = %33
  %46 = add <4 x i32> %42, %41
  %47 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %46)
  %48 = icmp eq i64 %32, %3
  br i1 %48, label %52, label %49

49:                                               ; preds = %29, %45
  %50 = phi i64 [ 0, %29 ], [ %32, %45 ]
  %51 = phi i32 [ 0, %29 ], [ %47, %45 ]
  br label %54

52:                                               ; preds = %54, %45, %1
  %53 = phi i32 [ 0, %1 ], [ %47, %45 ], [ %59, %54 ]
  ret i32 %53

54:                                               ; preds = %49, %54
  %55 = phi i64 [ %60, %54 ], [ %50, %49 ]
  %56 = phi i32 [ %59, %54 ], [ %51, %49 ]
  %57 = getelementptr inbounds nuw i32, ptr %4, i64 %55
  %58 = load i32, ptr %57, align 4, !tbaa !11
  %59 = add nsw i32 %58, %56
  %60 = add nuw nsw i64 %55, 1
  %61 = icmp eq i64 %60, %3
  br i1 %61, label %52, label %54, !llvm.loop !18
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #3

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS13MamaDebugTest", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!8, !8, i64 0}
!12 = distinct !{!12, !13, !14, !15}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.isvectorized", i32 1}
!15 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = distinct !{!16, !13, !15, !14}
!17 = distinct !{!17, !13, !14, !15}
!18 = distinct !{!18, !13, !15, !14}
