; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20090113-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20090113-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.descriptor_dimension = type { i32, i32, i32 }

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @msum_i4(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef readonly captures(none) %2) local_unnamed_addr #0 {
  %4 = alloca [7 x i32], align 4
  %5 = alloca [7 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  %6 = load i32, ptr %2, align 4, !tbaa !6
  %7 = add nsw i32 %6, -1
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %9 = sext i32 %7 to i64
  %10 = getelementptr inbounds %struct.descriptor_dimension, ptr %8, i64 %9
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %12 = load i32, ptr %11, align 4, !tbaa !10
  %13 = add i32 %12, 1
  %14 = getelementptr inbounds nuw i8, ptr %10, i64 4
  %15 = load i32, ptr %14, align 4, !tbaa !12
  %16 = sub i32 %13, %15
  %17 = icmp sgt i32 %6, 1
  br i1 %17, label %18, label %65

18:                                               ; preds = %3
  %19 = zext nneg i32 %7 to i64
  %20 = shl nuw nsw i64 %19, 2
  call void @llvm.memset.p0.i64(ptr nonnull align 4 %4, i8 0, i64 %20, i1 false), !tbaa !6
  %21 = icmp ult i32 %6, 10
  br i1 %21, label %48, label %22

22:                                               ; preds = %18
  %23 = and i64 %19, 7
  %24 = icmp eq i64 %23, 0
  %25 = select i1 %24, i64 8, i64 %23
  %26 = sub nsw i64 %19, %25
  br label %27

27:                                               ; preds = %27, %22
  %28 = phi i64 [ 0, %22 ], [ %46, %27 ]
  %29 = getelementptr inbounds nuw %struct.descriptor_dimension, ptr %8, i64 %28
  %30 = mul nuw nsw i64 %28, 12
  %31 = getelementptr inbounds nuw i8, ptr %8, i64 %30
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 4
  %33 = getelementptr inbounds nuw i8, ptr %29, i64 52
  %34 = load <11 x i32>, ptr %32, align 4, !tbaa !6
  %35 = shufflevector <11 x i32> %34, <11 x i32> poison, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
  %36 = shufflevector <11 x i32> %34, <11 x i32> poison, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
  %37 = load <11 x i32>, ptr %33, align 4, !tbaa !6
  %38 = shufflevector <11 x i32> %37, <11 x i32> poison, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
  %39 = shufflevector <11 x i32> %37, <11 x i32> poison, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
  %40 = add nsw <4 x i32> %36, splat (i32 1)
  %41 = add nsw <4 x i32> %39, splat (i32 1)
  %42 = sub <4 x i32> %40, %35
  %43 = sub <4 x i32> %41, %38
  %44 = getelementptr inbounds nuw i32, ptr %5, i64 %28
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 16
  store <4 x i32> %42, ptr %44, align 4, !tbaa !6
  store <4 x i32> %43, ptr %45, align 4, !tbaa !6
  %46 = add nuw i64 %28, 8
  %47 = icmp eq i64 %46, %26
  br i1 %47, label %48, label %27, !llvm.loop !13

48:                                               ; preds = %27, %18
  %49 = phi i64 [ 0, %18 ], [ %26, %27 ]
  br label %50

50:                                               ; preds = %48, %50
  %51 = phi i64 [ %60, %50 ], [ %49, %48 ]
  %52 = getelementptr inbounds nuw %struct.descriptor_dimension, ptr %8, i64 %51
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 8
  %54 = load i32, ptr %53, align 4, !tbaa !10
  %55 = add nsw i32 %54, 1
  %56 = getelementptr inbounds nuw i8, ptr %52, i64 4
  %57 = load i32, ptr %56, align 4, !tbaa !12
  %58 = sub i32 %55, %57
  %59 = getelementptr inbounds nuw i32, ptr %5, i64 %51
  store i32 %58, ptr %59, align 4, !tbaa !6
  %60 = add nuw nsw i64 %51, 1
  %61 = icmp eq i64 %60, %19
  br i1 %61, label %62, label %50, !llvm.loop !17

62:                                               ; preds = %50
  %63 = load i32, ptr %5, align 4, !tbaa !6
  %64 = load i32, ptr %4, align 4, !tbaa !6
  br label %65

65:                                               ; preds = %62, %3
  %66 = phi i32 [ %64, %62 ], [ undef, %3 ]
  %67 = phi i32 [ %63, %62 ], [ undef, %3 ]
  %68 = load ptr, ptr %0, align 8, !tbaa !18
  %69 = icmp sgt i32 %16, 0
  br i1 %69, label %76, label %70

70:                                               ; preds = %65
  %71 = xor i32 %66, -1
  %72 = add i32 %67, %71
  %73 = zext i32 %72 to i64
  %74 = shl nuw nsw i64 %73, 2
  %75 = add nuw nsw i64 %74, 4
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(1) %68, i8 0, i64 %75, i1 false), !tbaa !6
  br label %125

76:                                               ; preds = %65
  %77 = load ptr, ptr %1, align 8, !tbaa !18
  %78 = zext nneg i32 %16 to i64
  %79 = icmp ult i32 %16, 8
  %80 = and i64 %78, 2147483640
  %81 = trunc nuw nsw i64 %80 to i32
  %82 = shl nuw nsw i64 %80, 2
  %83 = icmp eq i64 %80, %78
  br label %84

84:                                               ; preds = %76, %119
  %85 = phi i32 [ %122, %119 ], [ %66, %76 ]
  %86 = phi ptr [ %121, %119 ], [ %77, %76 ]
  %87 = phi ptr [ %123, %119 ], [ %68, %76 ]
  br i1 %79, label %106, label %88

88:                                               ; preds = %84
  %89 = getelementptr i8, ptr %86, i64 %82
  br label %90

90:                                               ; preds = %90, %88
  %91 = phi i64 [ 0, %88 ], [ %101, %90 ]
  %92 = phi <4 x i32> [ zeroinitializer, %88 ], [ %99, %90 ]
  %93 = phi <4 x i32> [ zeroinitializer, %88 ], [ %100, %90 ]
  %94 = shl i64 %91, 2
  %95 = getelementptr i8, ptr %86, i64 %94
  %96 = getelementptr i8, ptr %95, i64 16
  %97 = load <4 x i32>, ptr %95, align 4, !tbaa !6
  %98 = load <4 x i32>, ptr %96, align 4, !tbaa !6
  %99 = add <4 x i32> %97, %92
  %100 = add <4 x i32> %98, %93
  %101 = add nuw i64 %91, 8
  %102 = icmp eq i64 %101, %80
  br i1 %102, label %103, label %90, !llvm.loop !22

103:                                              ; preds = %90
  %104 = add <4 x i32> %100, %99
  %105 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %104)
  br i1 %83, label %119, label %106

106:                                              ; preds = %84, %103
  %107 = phi i32 [ 0, %84 ], [ %105, %103 ]
  %108 = phi i32 [ 0, %84 ], [ %81, %103 ]
  %109 = phi ptr [ %86, %84 ], [ %89, %103 ]
  br label %110

110:                                              ; preds = %106, %110
  %111 = phi i32 [ %115, %110 ], [ %107, %106 ]
  %112 = phi i32 [ %116, %110 ], [ %108, %106 ]
  %113 = phi ptr [ %117, %110 ], [ %109, %106 ]
  %114 = load i32, ptr %113, align 4, !tbaa !6
  %115 = add nsw i32 %114, %111
  %116 = add nuw nsw i32 %112, 1
  %117 = getelementptr inbounds nuw i8, ptr %113, i64 4
  %118 = icmp eq i32 %116, %16
  br i1 %118, label %119, label %110, !llvm.loop !23

119:                                              ; preds = %110, %103
  %120 = phi i32 [ %105, %103 ], [ %115, %110 ]
  %121 = phi ptr [ %89, %103 ], [ %117, %110 ]
  store i32 %120, ptr %87, align 4, !tbaa !6
  %122 = add nsw i32 %85, 1
  store i32 %122, ptr %4, align 4, !tbaa !6
  %123 = getelementptr inbounds nuw i8, ptr %87, i64 4
  %124 = icmp eq i32 %122, %67
  br i1 %124, label %125, label %84, !llvm.loop !24

125:                                              ; preds = %119, %70
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #4

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree norecurse nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
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
!10 = !{!11, !7, i64 8}
!11 = !{!"descriptor_dimension", !7, i64 0, !7, i64 4, !7, i64 8}
!12 = !{!11, !7, i64 4}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !16, !15}
!18 = !{!19, !20, i64 0}
!19 = !{!"", !20, i64 0, !7, i64 8, !8, i64 12}
!20 = !{!"p1 int", !21, i64 0}
!21 = !{!"any pointer", !8, i64 0}
!22 = distinct !{!22, !14, !15, !16}
!23 = distinct !{!23, !14, !16, !15}
!24 = distinct !{!24, !14}
