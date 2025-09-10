; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr71083.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr71083.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.lock_chain = type { i32 }
%struct.lock_chain1 = type <{ i8, i16 }>

@test = dso_local global [101 x %struct.lock_chain] zeroinitializer, align 4
@test1 = dso_local global [101 x %struct.lock_chain1] zeroinitializer, align 1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local noundef ptr @foo(ptr noundef returned captures(ret: address, provenance) %0) local_unnamed_addr #0 {
  %2 = load i32, ptr %0, align 4
  %3 = and i32 %2, -256
  %4 = insertelement <4 x i32> poison, i32 %3, i64 0
  %5 = shufflevector <4 x i32> %4, <4 x i32> poison, <4 x i32> zeroinitializer
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %8 = load <4 x i32>, ptr %6, align 4
  %9 = load <4 x i32>, ptr %7, align 4
  %10 = and <4 x i32> %8, splat (i32 255)
  %11 = and <4 x i32> %9, splat (i32 255)
  %12 = or disjoint <4 x i32> %10, %5
  %13 = or disjoint <4 x i32> %11, %5
  store <4 x i32> %12, ptr %6, align 4
  store <4 x i32> %13, ptr %7, align 4
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 52
  %16 = load <4 x i32>, ptr %14, align 4
  %17 = load <4 x i32>, ptr %15, align 4
  %18 = and <4 x i32> %16, splat (i32 255)
  %19 = and <4 x i32> %17, splat (i32 255)
  %20 = or disjoint <4 x i32> %18, %5
  %21 = or disjoint <4 x i32> %19, %5
  store <4 x i32> %20, ptr %14, align 4
  store <4 x i32> %21, ptr %15, align 4
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 68
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 84
  %24 = load <4 x i32>, ptr %22, align 4
  %25 = load <4 x i32>, ptr %23, align 4
  %26 = and <4 x i32> %24, splat (i32 255)
  %27 = and <4 x i32> %25, splat (i32 255)
  %28 = or disjoint <4 x i32> %26, %5
  %29 = or disjoint <4 x i32> %27, %5
  store <4 x i32> %28, ptr %22, align 4
  store <4 x i32> %29, ptr %23, align 4
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 100
  %31 = getelementptr inbounds nuw i8, ptr %0, i64 116
  %32 = load <4 x i32>, ptr %30, align 4
  %33 = load <4 x i32>, ptr %31, align 4
  %34 = and <4 x i32> %32, splat (i32 255)
  %35 = and <4 x i32> %33, splat (i32 255)
  %36 = or disjoint <4 x i32> %34, %5
  %37 = or disjoint <4 x i32> %35, %5
  store <4 x i32> %36, ptr %30, align 4
  store <4 x i32> %37, ptr %31, align 4
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 132
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 148
  %40 = load <4 x i32>, ptr %38, align 4
  %41 = load <4 x i32>, ptr %39, align 4
  %42 = and <4 x i32> %40, splat (i32 255)
  %43 = and <4 x i32> %41, splat (i32 255)
  %44 = or disjoint <4 x i32> %42, %5
  %45 = or disjoint <4 x i32> %43, %5
  store <4 x i32> %44, ptr %38, align 4
  store <4 x i32> %45, ptr %39, align 4
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 164
  %47 = getelementptr inbounds nuw i8, ptr %0, i64 180
  %48 = load <4 x i32>, ptr %46, align 4
  %49 = load <4 x i32>, ptr %47, align 4
  %50 = and <4 x i32> %48, splat (i32 255)
  %51 = and <4 x i32> %49, splat (i32 255)
  %52 = or disjoint <4 x i32> %50, %5
  %53 = or disjoint <4 x i32> %51, %5
  store <4 x i32> %52, ptr %46, align 4
  store <4 x i32> %53, ptr %47, align 4
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 196
  %55 = getelementptr inbounds nuw i8, ptr %0, i64 212
  %56 = load <4 x i32>, ptr %54, align 4
  %57 = load <4 x i32>, ptr %55, align 4
  %58 = and <4 x i32> %56, splat (i32 255)
  %59 = and <4 x i32> %57, splat (i32 255)
  %60 = or disjoint <4 x i32> %58, %5
  %61 = or disjoint <4 x i32> %59, %5
  store <4 x i32> %60, ptr %54, align 4
  store <4 x i32> %61, ptr %55, align 4
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 228
  %63 = getelementptr inbounds nuw i8, ptr %0, i64 244
  %64 = load <4 x i32>, ptr %62, align 4
  %65 = load <4 x i32>, ptr %63, align 4
  %66 = and <4 x i32> %64, splat (i32 255)
  %67 = and <4 x i32> %65, splat (i32 255)
  %68 = or disjoint <4 x i32> %66, %5
  %69 = or disjoint <4 x i32> %67, %5
  store <4 x i32> %68, ptr %62, align 4
  store <4 x i32> %69, ptr %63, align 4
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 260
  %71 = getelementptr inbounds nuw i8, ptr %0, i64 276
  %72 = load <4 x i32>, ptr %70, align 4
  %73 = load <4 x i32>, ptr %71, align 4
  %74 = and <4 x i32> %72, splat (i32 255)
  %75 = and <4 x i32> %73, splat (i32 255)
  %76 = or disjoint <4 x i32> %74, %5
  %77 = or disjoint <4 x i32> %75, %5
  store <4 x i32> %76, ptr %70, align 4
  store <4 x i32> %77, ptr %71, align 4
  %78 = getelementptr inbounds nuw i8, ptr %0, i64 292
  %79 = getelementptr inbounds nuw i8, ptr %0, i64 308
  %80 = load <4 x i32>, ptr %78, align 4
  %81 = load <4 x i32>, ptr %79, align 4
  %82 = and <4 x i32> %80, splat (i32 255)
  %83 = and <4 x i32> %81, splat (i32 255)
  %84 = or disjoint <4 x i32> %82, %5
  %85 = or disjoint <4 x i32> %83, %5
  store <4 x i32> %84, ptr %78, align 4
  store <4 x i32> %85, ptr %79, align 4
  %86 = getelementptr inbounds nuw i8, ptr %0, i64 324
  %87 = getelementptr inbounds nuw i8, ptr %0, i64 340
  %88 = load <4 x i32>, ptr %86, align 4
  %89 = load <4 x i32>, ptr %87, align 4
  %90 = and <4 x i32> %88, splat (i32 255)
  %91 = and <4 x i32> %89, splat (i32 255)
  %92 = or disjoint <4 x i32> %90, %5
  %93 = or disjoint <4 x i32> %91, %5
  store <4 x i32> %92, ptr %86, align 4
  store <4 x i32> %93, ptr %87, align 4
  %94 = getelementptr inbounds nuw i8, ptr %0, i64 356
  %95 = getelementptr inbounds nuw i8, ptr %0, i64 372
  %96 = load <4 x i32>, ptr %94, align 4
  %97 = load <4 x i32>, ptr %95, align 4
  %98 = and <4 x i32> %96, splat (i32 255)
  %99 = and <4 x i32> %97, splat (i32 255)
  %100 = or disjoint <4 x i32> %98, %5
  %101 = or disjoint <4 x i32> %99, %5
  store <4 x i32> %100, ptr %94, align 4
  store <4 x i32> %101, ptr %95, align 4
  %102 = getelementptr inbounds nuw i8, ptr %0, i64 388
  %103 = load i32, ptr %102, align 4
  %104 = and i32 %103, 255
  %105 = or disjoint i32 %104, %3
  store i32 %105, ptr %102, align 4
  %106 = getelementptr inbounds nuw i8, ptr %0, i64 392
  %107 = load i32, ptr %106, align 4
  %108 = and i32 %107, 255
  %109 = or disjoint i32 %108, %3
  store i32 %109, ptr %106, align 4
  %110 = getelementptr inbounds nuw i8, ptr %0, i64 396
  %111 = load i32, ptr %110, align 4
  %112 = and i32 %111, 255
  %113 = or disjoint i32 %112, %3
  store i32 %113, ptr %110, align 4
  %114 = getelementptr inbounds nuw i8, ptr %0, i64 400
  %115 = load i32, ptr %114, align 4
  %116 = and i32 %115, 255
  %117 = or disjoint i32 %116, %3
  store i32 %117, ptr %114, align 4
  ret ptr %0
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local noundef ptr @bar(ptr noundef returned captures(ret: address, provenance) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %3 = load i16, ptr %2, align 1, !tbaa !6
  br label %4

4:                                                ; preds = %4, %1
  %5 = phi i64 [ 0, %1 ], [ %10, %4 ]
  %6 = or disjoint i64 %5, 1
  %7 = add nuw nsw i64 %5, 2
  %8 = getelementptr inbounds nuw %struct.lock_chain1, ptr %0, i64 %6, i32 1
  %9 = getelementptr inbounds nuw %struct.lock_chain1, ptr %0, i64 %7, i32 1
  store i16 %3, ptr %8, align 1, !tbaa !6
  store i16 %3, ptr %9, align 1, !tbaa !6
  %10 = add nuw i64 %5, 2
  %11 = icmp eq i64 %10, 100
  br i1 %11, label %12, label %4, !llvm.loop !11

12:                                               ; preds = %4
  ret ptr %0
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call ptr @foo(ptr noundef nonnull @test)
  %2 = tail call ptr @bar(ptr noundef nonnull @test1)
  ret i32 0
}

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !10, i64 1}
!7 = !{!"lock_chain1", !8, i64 0, !10, i64 1}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"short", !8, i64 0}
!11 = distinct !{!11, !12, !13, !14}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
