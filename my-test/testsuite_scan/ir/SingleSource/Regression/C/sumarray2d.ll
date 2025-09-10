; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/sumarray2d.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/sumarray2d.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [23 x i8] c"Sum(Array[%d,%d] = %d\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local i32 @SumArray(ptr noundef readonly captures(none) %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp eq i32 %1, 0
  %5 = icmp eq i32 %2, 0
  %6 = or i1 %4, %5
  br i1 %6, label %49, label %7

7:                                                ; preds = %3
  %8 = zext i32 %1 to i64
  %9 = zext i32 %2 to i64
  %10 = icmp ult i32 %2, 8
  %11 = and i64 %9, 4294967288
  %12 = icmp eq i64 %11, %9
  br label %13

13:                                               ; preds = %7, %45
  %14 = phi i64 [ 0, %7 ], [ %47, %45 ]
  %15 = phi i32 [ 0, %7 ], [ %46, %45 ]
  %16 = getelementptr inbounds nuw [100 x i32], ptr %0, i64 %14
  br i1 %10, label %34, label %17

17:                                               ; preds = %13
  %18 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %15, i64 0
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %29, %19 ]
  %21 = phi <4 x i32> [ %18, %17 ], [ %27, %19 ]
  %22 = phi <4 x i32> [ zeroinitializer, %17 ], [ %28, %19 ]
  %23 = getelementptr inbounds nuw i32, ptr %16, i64 %20
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !6
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !6
  %27 = add <4 x i32> %25, %21
  %28 = add <4 x i32> %26, %22
  %29 = add nuw i64 %20, 8
  %30 = icmp eq i64 %29, %11
  br i1 %30, label %31, label %19, !llvm.loop !10

31:                                               ; preds = %19
  %32 = add <4 x i32> %28, %27
  %33 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %32)
  br i1 %12, label %45, label %34

34:                                               ; preds = %13, %31
  %35 = phi i64 [ 0, %13 ], [ %11, %31 ]
  %36 = phi i32 [ %15, %13 ], [ %33, %31 ]
  br label %37

37:                                               ; preds = %34, %37
  %38 = phi i64 [ %43, %37 ], [ %35, %34 ]
  %39 = phi i32 [ %42, %37 ], [ %36, %34 ]
  %40 = getelementptr inbounds nuw i32, ptr %16, i64 %38
  %41 = load i32, ptr %40, align 4, !tbaa !6
  %42 = add nsw i32 %41, %39
  %43 = add nuw nsw i64 %38, 1
  %44 = icmp eq i64 %43, %9
  br i1 %44, label %45, label %37, !llvm.loop !14

45:                                               ; preds = %37, %31
  %46 = phi i32 [ %33, %31 ], [ %42, %37 ]
  %47 = add nuw nsw i64 %14, 1
  %48 = icmp eq i64 %47, %8
  br i1 %48, label %49, label %13, !llvm.loop !15

49:                                               ; preds = %45, %3
  %50 = phi i32 [ 0, %3 ], [ %46, %45 ]
  ret i32 %50
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [100 x [100 x i32]], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  br label %2

2:                                                ; preds = %2, %0
  %3 = phi i64 [ 0, %0 ], [ %12, %2 ]
  %4 = or disjoint i64 %3, 1
  %5 = trunc i64 %3 to i32
  %6 = xor i32 %5, -1
  %7 = getelementptr inbounds nuw [100 x i32], ptr %1, i64 %3
  %8 = getelementptr inbounds nuw [100 x i32], ptr %1, i64 %4
  %9 = getelementptr inbounds nuw i32, ptr %7, i64 %3
  %10 = getelementptr inbounds nuw i32, ptr %8, i64 %4
  %11 = sub i32 0, %5
  store i32 %11, ptr %9, align 4, !tbaa !6
  store i32 %6, ptr %10, align 4, !tbaa !6
  %12 = add nuw i64 %3, 2
  %13 = icmp eq i64 %12, 100
  br i1 %13, label %14, label %2, !llvm.loop !16

14:                                               ; preds = %2, %27
  %15 = phi i64 [ %28, %27 ], [ 0, %2 ]
  %16 = getelementptr inbounds nuw [100 x i32], ptr %1, i64 %15
  br label %17

17:                                               ; preds = %14, %24
  %18 = phi i64 [ 0, %14 ], [ %25, %24 ]
  %19 = icmp eq i64 %18, %15
  br i1 %19, label %24, label %20

20:                                               ; preds = %17
  %21 = add nuw nsw i64 %18, %15
  %22 = getelementptr inbounds nuw i32, ptr %16, i64 %18
  %23 = trunc nuw nsw i64 %21 to i32
  store i32 %23, ptr %22, align 4, !tbaa !6
  br label %24

24:                                               ; preds = %17, %20
  %25 = add nuw nsw i64 %18, 1
  %26 = icmp eq i64 %25, 100
  br i1 %26, label %27, label %17, !llvm.loop !17

27:                                               ; preds = %24
  %28 = add nuw nsw i64 %15, 1
  %29 = icmp eq i64 %28, 100
  br i1 %29, label %30, label %14, !llvm.loop !18

30:                                               ; preds = %27, %30
  %31 = phi i64 [ %119, %30 ], [ 0, %27 ]
  %32 = phi i32 [ %118, %30 ], [ 0, %27 ]
  %33 = getelementptr inbounds nuw [100 x i32], ptr %1, i64 %31
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 368
  %35 = load <4 x i32>, ptr %34, align 4, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %33, i64 336
  %37 = load <4 x i32>, ptr %36, align 4, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %33, i64 304
  %39 = load <4 x i32>, ptr %38, align 4, !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %33, i64 272
  %41 = load <4 x i32>, ptr %40, align 4, !tbaa !6
  %42 = getelementptr inbounds nuw i8, ptr %33, i64 240
  %43 = load <4 x i32>, ptr %42, align 4, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %33, i64 208
  %45 = load <4 x i32>, ptr %44, align 4, !tbaa !6
  %46 = getelementptr inbounds nuw i8, ptr %33, i64 176
  %47 = load <4 x i32>, ptr %46, align 4, !tbaa !6
  %48 = getelementptr inbounds nuw i8, ptr %33, i64 144
  %49 = load <4 x i32>, ptr %48, align 4, !tbaa !6
  %50 = getelementptr inbounds nuw i8, ptr %33, i64 112
  %51 = load <4 x i32>, ptr %50, align 4, !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %33, i64 80
  %53 = load <4 x i32>, ptr %52, align 4, !tbaa !6
  %54 = getelementptr inbounds nuw i8, ptr %33, i64 48
  %55 = load <4 x i32>, ptr %54, align 4, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %33, i64 16
  %57 = load <4 x i32>, ptr %56, align 4, !tbaa !6
  %58 = add <4 x i32> %55, %57
  %59 = add <4 x i32> %53, %58
  %60 = add <4 x i32> %51, %59
  %61 = add <4 x i32> %49, %60
  %62 = add <4 x i32> %47, %61
  %63 = add <4 x i32> %45, %62
  %64 = add <4 x i32> %43, %63
  %65 = add <4 x i32> %41, %64
  %66 = add <4 x i32> %39, %65
  %67 = add <4 x i32> %37, %66
  %68 = add <4 x i32> %35, %67
  %69 = getelementptr inbounds nuw i8, ptr %33, i64 352
  %70 = load <4 x i32>, ptr %69, align 4, !tbaa !6
  %71 = getelementptr inbounds nuw i8, ptr %33, i64 320
  %72 = load <4 x i32>, ptr %71, align 4, !tbaa !6
  %73 = getelementptr inbounds nuw i8, ptr %33, i64 288
  %74 = load <4 x i32>, ptr %73, align 4, !tbaa !6
  %75 = getelementptr inbounds nuw i8, ptr %33, i64 256
  %76 = load <4 x i32>, ptr %75, align 4, !tbaa !6
  %77 = getelementptr inbounds nuw i8, ptr %33, i64 224
  %78 = load <4 x i32>, ptr %77, align 4, !tbaa !6
  %79 = getelementptr inbounds nuw i8, ptr %33, i64 192
  %80 = load <4 x i32>, ptr %79, align 4, !tbaa !6
  %81 = getelementptr inbounds nuw i8, ptr %33, i64 160
  %82 = load <4 x i32>, ptr %81, align 4, !tbaa !6
  %83 = getelementptr inbounds nuw i8, ptr %33, i64 128
  %84 = load <4 x i32>, ptr %83, align 4, !tbaa !6
  %85 = getelementptr inbounds nuw i8, ptr %33, i64 96
  %86 = load <4 x i32>, ptr %85, align 4, !tbaa !6
  %87 = getelementptr inbounds nuw i8, ptr %33, i64 64
  %88 = load <4 x i32>, ptr %87, align 4, !tbaa !6
  %89 = getelementptr inbounds nuw i8, ptr %33, i64 32
  %90 = load <4 x i32>, ptr %89, align 4, !tbaa !6
  %91 = load <4 x i32>, ptr %33, align 4, !tbaa !6
  %92 = insertelement <4 x i32> <i32 poison, i32 0, i32 0, i32 0>, i32 %32, i64 0
  %93 = add <4 x i32> %91, %92
  %94 = add <4 x i32> %90, %93
  %95 = add <4 x i32> %88, %94
  %96 = add <4 x i32> %86, %95
  %97 = add <4 x i32> %84, %96
  %98 = add <4 x i32> %82, %97
  %99 = add <4 x i32> %80, %98
  %100 = add <4 x i32> %78, %99
  %101 = add <4 x i32> %76, %100
  %102 = add <4 x i32> %74, %101
  %103 = add <4 x i32> %72, %102
  %104 = add <4 x i32> %70, %103
  %105 = add <4 x i32> %68, %104
  %106 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %105)
  %107 = getelementptr inbounds nuw i8, ptr %33, i64 384
  %108 = load i32, ptr %107, align 4, !tbaa !6
  %109 = add nsw i32 %108, %106
  %110 = getelementptr inbounds nuw i8, ptr %33, i64 388
  %111 = load i32, ptr %110, align 4, !tbaa !6
  %112 = add nsw i32 %111, %109
  %113 = getelementptr inbounds nuw i8, ptr %33, i64 392
  %114 = load i32, ptr %113, align 4, !tbaa !6
  %115 = add nsw i32 %114, %112
  %116 = getelementptr inbounds nuw i8, ptr %33, i64 396
  %117 = load i32, ptr %116, align 4, !tbaa !6
  %118 = add nsw i32 %117, %115
  %119 = add nuw nsw i64 %31, 1
  %120 = icmp eq i64 %119, 100
  br i1 %120, label %121, label %30, !llvm.loop !15

121:                                              ; preds = %30
  %122 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 100, i32 noundef 100, i32 noundef %118)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }

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
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !13, !12}
!15 = distinct !{!15, !11}
!16 = distinct !{!16, !11, !12, !13}
!17 = distinct !{!17, !11}
!18 = distinct !{!18, !11}
