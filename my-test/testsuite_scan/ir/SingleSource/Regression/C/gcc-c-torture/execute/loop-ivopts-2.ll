; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/loop-ivopts-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/loop-ivopts-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local void @check(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  br label %5

2:                                                ; preds = %5
  %3 = add nuw nsw i64 %6, 1
  %4 = icmp eq i64 %3, 288
  br i1 %4, label %20, label %5, !llvm.loop !6

5:                                                ; preds = %1, %2
  %6 = phi i64 [ 0, %1 ], [ %3, %2 ]
  %7 = getelementptr inbounds nuw i32, ptr %0, i64 %6
  %8 = load i32, ptr %7, align 4, !tbaa !8
  %9 = trunc i64 %6 to i32
  %10 = add i32 %9, -280
  %11 = icmp ult i32 %10, -24
  %12 = select i1 %11, i32 8, i32 7
  %13 = trunc i64 %6 to i32
  %14 = add i32 %13, -144
  %15 = icmp ult i32 %14, 112
  %16 = zext i1 %15 to i32
  %17 = add nuw nsw i32 %12, %16
  %18 = icmp eq i32 %8, %17
  br i1 %18, label %2, label %19

19:                                               ; preds = %5
  tail call void @abort() #3
  unreachable

20:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [288 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  store <4 x i32> splat (i32 8), ptr %1, align 16, !tbaa !8
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store <4 x i32> splat (i32 8), ptr %2, align 16, !tbaa !8
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store <4 x i32> splat (i32 8), ptr %3, align 16, !tbaa !8
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store <4 x i32> splat (i32 8), ptr %4, align 16, !tbaa !8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 64
  store <4 x i32> splat (i32 8), ptr %5, align 16, !tbaa !8
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 80
  store <4 x i32> splat (i32 8), ptr %6, align 16, !tbaa !8
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 96
  store <4 x i32> splat (i32 8), ptr %7, align 16, !tbaa !8
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 112
  store <4 x i32> splat (i32 8), ptr %8, align 16, !tbaa !8
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 128
  store <4 x i32> splat (i32 8), ptr %9, align 16, !tbaa !8
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 144
  store <4 x i32> splat (i32 8), ptr %10, align 16, !tbaa !8
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 160
  store <4 x i32> splat (i32 8), ptr %11, align 16, !tbaa !8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 176
  store <4 x i32> splat (i32 8), ptr %12, align 16, !tbaa !8
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 192
  store <4 x i32> splat (i32 8), ptr %13, align 16, !tbaa !8
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 208
  store <4 x i32> splat (i32 8), ptr %14, align 16, !tbaa !8
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 224
  store <4 x i32> splat (i32 8), ptr %15, align 16, !tbaa !8
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 240
  store <4 x i32> splat (i32 8), ptr %16, align 16, !tbaa !8
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 256
  store <4 x i32> splat (i32 8), ptr %17, align 16, !tbaa !8
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 272
  store <4 x i32> splat (i32 8), ptr %18, align 16, !tbaa !8
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 288
  store <4 x i32> splat (i32 8), ptr %19, align 16, !tbaa !8
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 304
  store <4 x i32> splat (i32 8), ptr %20, align 16, !tbaa !8
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 320
  store <4 x i32> splat (i32 8), ptr %21, align 16, !tbaa !8
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 336
  store <4 x i32> splat (i32 8), ptr %22, align 16, !tbaa !8
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 352
  store <4 x i32> splat (i32 8), ptr %23, align 16, !tbaa !8
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 368
  store <4 x i32> splat (i32 8), ptr %24, align 16, !tbaa !8
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 384
  store <4 x i32> splat (i32 8), ptr %25, align 16, !tbaa !8
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 400
  store <4 x i32> splat (i32 8), ptr %26, align 16, !tbaa !8
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 416
  store <4 x i32> splat (i32 8), ptr %27, align 16, !tbaa !8
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 432
  store <4 x i32> splat (i32 8), ptr %28, align 16, !tbaa !8
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 448
  store <4 x i32> splat (i32 8), ptr %29, align 16, !tbaa !8
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 464
  store <4 x i32> splat (i32 8), ptr %30, align 16, !tbaa !8
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 480
  store <4 x i32> splat (i32 8), ptr %31, align 16, !tbaa !8
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 496
  store <4 x i32> splat (i32 8), ptr %32, align 16, !tbaa !8
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 512
  store <4 x i32> splat (i32 8), ptr %33, align 16, !tbaa !8
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 528
  store <4 x i32> splat (i32 8), ptr %34, align 16, !tbaa !8
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 544
  store <4 x i32> splat (i32 8), ptr %35, align 16, !tbaa !8
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 560
  store <4 x i32> splat (i32 8), ptr %36, align 16, !tbaa !8
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 576
  store <4 x i32> splat (i32 9), ptr %37, align 16, !tbaa !8
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 592
  store <4 x i32> splat (i32 9), ptr %38, align 16, !tbaa !8
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 608
  store <4 x i32> splat (i32 9), ptr %39, align 16, !tbaa !8
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 624
  store <4 x i32> splat (i32 9), ptr %40, align 16, !tbaa !8
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 640
  store <4 x i32> splat (i32 9), ptr %41, align 16, !tbaa !8
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 656
  store <4 x i32> splat (i32 9), ptr %42, align 16, !tbaa !8
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 672
  store <4 x i32> splat (i32 9), ptr %43, align 16, !tbaa !8
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 688
  store <4 x i32> splat (i32 9), ptr %44, align 16, !tbaa !8
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 704
  store <4 x i32> splat (i32 9), ptr %45, align 16, !tbaa !8
  %46 = getelementptr inbounds nuw i8, ptr %1, i64 720
  store <4 x i32> splat (i32 9), ptr %46, align 16, !tbaa !8
  %47 = getelementptr inbounds nuw i8, ptr %1, i64 736
  store <4 x i32> splat (i32 9), ptr %47, align 16, !tbaa !8
  %48 = getelementptr inbounds nuw i8, ptr %1, i64 752
  store <4 x i32> splat (i32 9), ptr %48, align 16, !tbaa !8
  %49 = getelementptr inbounds nuw i8, ptr %1, i64 768
  store <4 x i32> splat (i32 9), ptr %49, align 16, !tbaa !8
  %50 = getelementptr inbounds nuw i8, ptr %1, i64 784
  store <4 x i32> splat (i32 9), ptr %50, align 16, !tbaa !8
  %51 = getelementptr inbounds nuw i8, ptr %1, i64 800
  store <4 x i32> splat (i32 9), ptr %51, align 16, !tbaa !8
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 816
  store <4 x i32> splat (i32 9), ptr %52, align 16, !tbaa !8
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 832
  store <4 x i32> splat (i32 9), ptr %53, align 16, !tbaa !8
  %54 = getelementptr inbounds nuw i8, ptr %1, i64 848
  store <4 x i32> splat (i32 9), ptr %54, align 16, !tbaa !8
  %55 = getelementptr inbounds nuw i8, ptr %1, i64 864
  store <4 x i32> splat (i32 9), ptr %55, align 16, !tbaa !8
  %56 = getelementptr inbounds nuw i8, ptr %1, i64 880
  store <4 x i32> splat (i32 9), ptr %56, align 16, !tbaa !8
  %57 = getelementptr inbounds nuw i8, ptr %1, i64 896
  store <4 x i32> splat (i32 9), ptr %57, align 16, !tbaa !8
  %58 = getelementptr inbounds nuw i8, ptr %1, i64 912
  store <4 x i32> splat (i32 9), ptr %58, align 16, !tbaa !8
  %59 = getelementptr inbounds nuw i8, ptr %1, i64 928
  store <4 x i32> splat (i32 9), ptr %59, align 16, !tbaa !8
  %60 = getelementptr inbounds nuw i8, ptr %1, i64 944
  store <4 x i32> splat (i32 9), ptr %60, align 16, !tbaa !8
  %61 = getelementptr inbounds nuw i8, ptr %1, i64 960
  store <4 x i32> splat (i32 9), ptr %61, align 16, !tbaa !8
  %62 = getelementptr inbounds nuw i8, ptr %1, i64 976
  store <4 x i32> splat (i32 9), ptr %62, align 16, !tbaa !8
  %63 = getelementptr inbounds nuw i8, ptr %1, i64 992
  store <4 x i32> splat (i32 9), ptr %63, align 16, !tbaa !8
  %64 = getelementptr inbounds nuw i8, ptr %1, i64 1008
  store <4 x i32> splat (i32 9), ptr %64, align 16, !tbaa !8
  %65 = getelementptr inbounds nuw i8, ptr %1, i64 1024
  store <4 x i32> splat (i32 7), ptr %65, align 16, !tbaa !8
  %66 = getelementptr inbounds nuw i8, ptr %1, i64 1040
  store <4 x i32> splat (i32 7), ptr %66, align 16, !tbaa !8
  %67 = getelementptr inbounds nuw i8, ptr %1, i64 1056
  store <4 x i32> splat (i32 7), ptr %67, align 16, !tbaa !8
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 1072
  store <4 x i32> splat (i32 7), ptr %68, align 16, !tbaa !8
  %69 = getelementptr inbounds nuw i8, ptr %1, i64 1088
  store <4 x i32> splat (i32 7), ptr %69, align 16, !tbaa !8
  %70 = getelementptr inbounds nuw i8, ptr %1, i64 1104
  store <4 x i32> splat (i32 7), ptr %70, align 16, !tbaa !8
  %71 = getelementptr inbounds nuw i8, ptr %1, i64 1120
  store <4 x i32> splat (i32 8), ptr %71, align 16, !tbaa !8
  %72 = getelementptr inbounds nuw i8, ptr %1, i64 1136
  store <4 x i32> splat (i32 8), ptr %72, align 16, !tbaa !8
  br label %73

73:                                               ; preds = %73, %0
  %74 = phi i64 [ 0, %0 ], [ %87, %73 ]
  %75 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %0 ], [ %92, %73 ]
  %76 = getelementptr inbounds nuw i32, ptr %1, i64 %74
  %77 = load <4 x i32>, ptr %76, align 16, !tbaa !8
  %78 = add <4 x i32> %75, splat (i32 -280)
  %79 = icmp ult <4 x i32> %78, splat (i32 -24)
  %80 = select <4 x i1> %79, <4 x i32> splat (i32 8), <4 x i32> splat (i32 7)
  %81 = add <4 x i32> %75, splat (i32 -144)
  %82 = icmp ult <4 x i32> %81, splat (i32 112)
  %83 = zext <4 x i1> %82 to <4 x i32>
  %84 = add nuw nsw <4 x i32> %80, %83
  %85 = freeze <4 x i32> %77
  %86 = icmp ne <4 x i32> %85, %84
  %87 = add nuw i64 %74, 4
  %88 = bitcast <4 x i1> %86 to i4
  %89 = icmp ne i4 %88, 0
  %90 = icmp eq i64 %87, 288
  %91 = or i1 %89, %90
  %92 = add <4 x i32> %75, splat (i32 4)
  br i1 %91, label %93, label %73, !llvm.loop !12

93:                                               ; preds = %73
  br i1 %89, label %94, label %95

94:                                               ; preds = %93
  tail call void @abort() #3
  unreachable

95:                                               ; preds = %93
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = distinct !{!12, !7, !13, !14}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
