; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/sumarray.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/sumarray.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"Produced: %d\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local i32 @SumArray(ptr noundef captures(none) initializes((136, 140)) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 136
  store i32 1234, ptr %3, align 4, !tbaa !6
  %4 = icmp eq i32 %1, 0
  br i1 %4, label %37, label %5

5:                                                ; preds = %2
  %6 = zext i32 %1 to i64
  %7 = icmp ult i32 %1, 8
  br i1 %7, label %26, label %8

8:                                                ; preds = %5
  %9 = and i64 %6, 4294967288
  br label %10

10:                                               ; preds = %10, %8
  %11 = phi i64 [ 0, %8 ], [ %20, %10 ]
  %12 = phi <4 x i32> [ zeroinitializer, %8 ], [ %18, %10 ]
  %13 = phi <4 x i32> [ zeroinitializer, %8 ], [ %19, %10 ]
  %14 = getelementptr inbounds nuw i32, ptr %0, i64 %11
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %16 = load <4 x i32>, ptr %14, align 4, !tbaa !6
  %17 = load <4 x i32>, ptr %15, align 4, !tbaa !6
  %18 = add <4 x i32> %16, %12
  %19 = add <4 x i32> %17, %13
  %20 = add nuw i64 %11, 8
  %21 = icmp eq i64 %20, %9
  br i1 %21, label %22, label %10, !llvm.loop !10

22:                                               ; preds = %10
  %23 = add <4 x i32> %19, %18
  %24 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %23)
  %25 = icmp eq i64 %9, %6
  br i1 %25, label %37, label %26

26:                                               ; preds = %5, %22
  %27 = phi i64 [ 0, %5 ], [ %9, %22 ]
  %28 = phi i32 [ 0, %5 ], [ %24, %22 ]
  br label %29

29:                                               ; preds = %26, %29
  %30 = phi i64 [ %35, %29 ], [ %27, %26 ]
  %31 = phi i32 [ %34, %29 ], [ %28, %26 ]
  %32 = getelementptr inbounds nuw i32, ptr %0, i64 %30
  %33 = load i32, ptr %32, align 4, !tbaa !6
  %34 = add i32 %33, %31
  %35 = add nuw nsw i64 %30, 1
  %36 = icmp eq i64 %35, %6
  br i1 %36, label %37, label %29, !llvm.loop !14

37:                                               ; preds = %29, %22, %2
  %38 = phi i32 [ 0, %2 ], [ %24, %22 ], [ %34, %29 ]
  ret i32 %38
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call noalias dereferenceable_or_null(400) ptr @malloc(i64 noundef 400) #5
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 384
  store <4 x i32> <i32 0, i32 2, i32 8, i32 6>, ptr %1, align 4, !tbaa !6
  store <4 x i32> <i32 16, i32 10, i32 24, i32 14>, ptr %2, align 4, !tbaa !6
  store <4 x i32> <i32 32, i32 18, i32 40, i32 22>, ptr %3, align 4, !tbaa !6
  store <4 x i32> <i32 48, i32 26, i32 56, i32 30>, ptr %4, align 4, !tbaa !6
  store <4 x i32> <i32 64, i32 34, i32 72, i32 38>, ptr %5, align 4, !tbaa !6
  store <4 x i32> <i32 80, i32 42, i32 88, i32 46>, ptr %6, align 4, !tbaa !6
  store <4 x i32> <i32 96, i32 50, i32 104, i32 54>, ptr %7, align 4, !tbaa !6
  store <4 x i32> <i32 112, i32 58, i32 120, i32 62>, ptr %8, align 4, !tbaa !6
  store <4 x i32> <i32 144, i32 74, i32 152, i32 78>, ptr %10, align 4, !tbaa !6
  store <4 x i32> <i32 160, i32 82, i32 168, i32 86>, ptr %11, align 4, !tbaa !6
  store <4 x i32> <i32 176, i32 90, i32 184, i32 94>, ptr %12, align 4, !tbaa !6
  store <4 x i32> <i32 192, i32 98, i32 200, i32 102>, ptr %13, align 4, !tbaa !6
  store <4 x i32> <i32 208, i32 106, i32 216, i32 110>, ptr %14, align 4, !tbaa !6
  store <4 x i32> <i32 224, i32 114, i32 232, i32 118>, ptr %15, align 4, !tbaa !6
  store <4 x i32> <i32 240, i32 122, i32 248, i32 126>, ptr %16, align 4, !tbaa !6
  store <4 x i32> <i32 256, i32 130, i32 264, i32 134>, ptr %17, align 4, !tbaa !6
  store <4 x i32> <i32 272, i32 138, i32 280, i32 142>, ptr %18, align 4, !tbaa !6
  store <4 x i32> <i32 288, i32 146, i32 296, i32 150>, ptr %19, align 4, !tbaa !6
  store <4 x i32> <i32 304, i32 154, i32 312, i32 158>, ptr %20, align 4, !tbaa !6
  store <4 x i32> <i32 320, i32 162, i32 328, i32 166>, ptr %21, align 4, !tbaa !6
  store <4 x i32> <i32 336, i32 170, i32 344, i32 174>, ptr %22, align 4, !tbaa !6
  store <4 x i32> <i32 352, i32 178, i32 360, i32 182>, ptr %23, align 4, !tbaa !6
  store <4 x i32> <i32 368, i32 186, i32 376, i32 190>, ptr %24, align 4, !tbaa !6
  store <4 x i32> <i32 384, i32 194, i32 392, i32 198>, ptr %25, align 4, !tbaa !6
  store <4 x i32> <i32 128, i32 66, i32 1234, i32 70>, ptr %9, align 4, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 368
  %27 = load <4 x i32>, ptr %26, align 4, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 336
  %29 = load <4 x i32>, ptr %28, align 4, !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 304
  %31 = load <4 x i32>, ptr %30, align 4, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 272
  %33 = load <4 x i32>, ptr %32, align 4, !tbaa !6
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %35 = load <4 x i32>, ptr %34, align 4, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %37 = load <4 x i32>, ptr %36, align 4, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %39 = load <4 x i32>, ptr %38, align 4, !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %41 = load <4 x i32>, ptr %40, align 4, !tbaa !6
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %43 = load <4 x i32>, ptr %42, align 4, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %45 = load <4 x i32>, ptr %44, align 4, !tbaa !6
  %46 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %47 = load <4 x i32>, ptr %46, align 4, !tbaa !6
  %48 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %49 = load <4 x i32>, ptr %48, align 4, !tbaa !6
  %50 = add <4 x i32> %47, %49
  %51 = add <4 x i32> %45, %50
  %52 = add <4 x i32> %43, %51
  %53 = add <4 x i32> %41, %52
  %54 = add <4 x i32> %39, %53
  %55 = add <4 x i32> %37, %54
  %56 = add <4 x i32> %35, %55
  %57 = add <4 x i32> %33, %56
  %58 = add <4 x i32> %31, %57
  %59 = add <4 x i32> %29, %58
  %60 = add <4 x i32> %27, %59
  %61 = getelementptr inbounds nuw i8, ptr %1, i64 352
  %62 = load <4 x i32>, ptr %61, align 4, !tbaa !6
  %63 = getelementptr inbounds nuw i8, ptr %1, i64 320
  %64 = load <4 x i32>, ptr %63, align 4, !tbaa !6
  %65 = getelementptr inbounds nuw i8, ptr %1, i64 288
  %66 = load <4 x i32>, ptr %65, align 4, !tbaa !6
  %67 = getelementptr inbounds nuw i8, ptr %1, i64 256
  %68 = load <4 x i32>, ptr %67, align 4, !tbaa !6
  %69 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %70 = load <4 x i32>, ptr %69, align 4, !tbaa !6
  %71 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %72 = load <4 x i32>, ptr %71, align 4, !tbaa !6
  %73 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %74 = load <4 x i32>, ptr %73, align 4, !tbaa !6
  %75 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %76 = load <4 x i32>, ptr %75, align 4, !tbaa !6
  %77 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %78 = load <4 x i32>, ptr %77, align 4, !tbaa !6
  %79 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %80 = load <4 x i32>, ptr %79, align 4, !tbaa !6
  %81 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %82 = load <4 x i32>, ptr %81, align 4, !tbaa !6
  %83 = load <4 x i32>, ptr %1, align 4, !tbaa !6
  %84 = add <4 x i32> %82, %83
  %85 = add <4 x i32> %80, %84
  %86 = add <4 x i32> %78, %85
  %87 = add <4 x i32> %76, %86
  %88 = add <4 x i32> %74, %87
  %89 = add <4 x i32> %72, %88
  %90 = add <4 x i32> %70, %89
  %91 = add <4 x i32> %68, %90
  %92 = add <4 x i32> %66, %91
  %93 = add <4 x i32> %64, %92
  %94 = add <4 x i32> %62, %93
  %95 = add <4 x i32> %60, %94
  %96 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %95)
  %97 = getelementptr inbounds nuw i8, ptr %1, i64 384
  %98 = load i32, ptr %97, align 4, !tbaa !6
  %99 = add i32 %98, %96
  %100 = getelementptr inbounds nuw i8, ptr %1, i64 388
  %101 = load i32, ptr %100, align 4, !tbaa !6
  %102 = add i32 %101, %99
  %103 = getelementptr inbounds nuw i8, ptr %1, i64 392
  %104 = load i32, ptr %103, align 4, !tbaa !6
  %105 = add i32 %104, %102
  %106 = getelementptr inbounds nuw i8, ptr %1, i64 396
  %107 = load i32, ptr %106, align 4, !tbaa !6
  %108 = add i32 %107, %105
  %109 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %108)
  ret i32 0
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind allocsize(0) }

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
