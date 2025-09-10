; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr37573.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr37573.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { ptr, i32, [624 x i32] }

@p = internal global [23 x i8] c"\C0I\172b\1E.\D5L\19(I\91\E4r\83\91=\93\83\B3a8", align 16
@q = internal global [23 x i8] c">AUTOIT UNICODE SCRIPT<", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call fastcc void @bar()
  %1 = tail call i32 @bcmp(ptr noundef nonnull dereferenceable(23) @p, ptr noundef nonnull dereferenceable(23) @q, i64 23)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #6
  unreachable

4:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define internal fastcc void @bar() unnamed_addr #2 {
  %1 = alloca %struct.S, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 41589, ptr %2, align 4, !tbaa !6
  br label %3

3:                                                ; preds = %0, %3
  %4 = phi i32 [ 41589, %0 ], [ %10, %3 ]
  %5 = phi i64 [ 1, %0 ], [ %12, %3 ]
  %6 = lshr i32 %4, 30
  %7 = xor i32 %6, %4
  %8 = mul i32 %7, 1812433253
  %9 = trunc nuw nsw i64 %5 to i32
  %10 = add i32 %8, %9
  %11 = getelementptr inbounds nuw i32, ptr %2, i64 %5
  store i32 %10, ptr %11, align 4, !tbaa !6
  %12 = add nuw nsw i64 %5, 1
  %13 = icmp eq i64 %12, 624
  br i1 %13, label %14, label %3, !llvm.loop !10

14:                                               ; preds = %3
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 1, ptr %15, align 8, !tbaa !12
  %16 = call fastcc i8 @foo(ptr noundef %1)
  %17 = call fastcc i8 @foo(ptr noundef %1)
  %18 = call fastcc i8 @foo(ptr noundef %1)
  %19 = call fastcc i8 @foo(ptr noundef %1)
  %20 = call fastcc i8 @foo(ptr noundef %1)
  %21 = call fastcc i8 @foo(ptr noundef %1)
  %22 = call fastcc i8 @foo(ptr noundef %1)
  %23 = call fastcc i8 @foo(ptr noundef %1)
  %24 = call fastcc i8 @foo(ptr noundef %1)
  %25 = call fastcc i8 @foo(ptr noundef %1)
  %26 = call fastcc i8 @foo(ptr noundef %1)
  %27 = call fastcc i8 @foo(ptr noundef %1)
  %28 = call fastcc i8 @foo(ptr noundef %1)
  %29 = call fastcc i8 @foo(ptr noundef %1)
  %30 = call fastcc i8 @foo(ptr noundef %1)
  %31 = call fastcc i8 @foo(ptr noundef %1)
  %32 = load <16 x i8>, ptr @p, align 16, !tbaa !16
  %33 = insertelement <16 x i8> poison, i8 %16, i64 0
  %34 = insertelement <16 x i8> %33, i8 %17, i64 1
  %35 = insertelement <16 x i8> %34, i8 %18, i64 2
  %36 = insertelement <16 x i8> %35, i8 %19, i64 3
  %37 = insertelement <16 x i8> %36, i8 %20, i64 4
  %38 = insertelement <16 x i8> %37, i8 %21, i64 5
  %39 = insertelement <16 x i8> %38, i8 %22, i64 6
  %40 = insertelement <16 x i8> %39, i8 %23, i64 7
  %41 = insertelement <16 x i8> %40, i8 %24, i64 8
  %42 = insertelement <16 x i8> %41, i8 %25, i64 9
  %43 = insertelement <16 x i8> %42, i8 %26, i64 10
  %44 = insertelement <16 x i8> %43, i8 %27, i64 11
  %45 = insertelement <16 x i8> %44, i8 %28, i64 12
  %46 = insertelement <16 x i8> %45, i8 %29, i64 13
  %47 = insertelement <16 x i8> %46, i8 %30, i64 14
  %48 = insertelement <16 x i8> %47, i8 %31, i64 15
  %49 = xor <16 x i8> %32, %48
  store <16 x i8> %49, ptr @p, align 16, !tbaa !16
  %50 = call fastcc i8 @foo(ptr noundef %1)
  %51 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 16), align 16, !tbaa !16
  %52 = xor i8 %51, %50
  store i8 %52, ptr getelementptr inbounds nuw (i8, ptr @p, i64 16), align 16, !tbaa !16
  %53 = call fastcc i8 @foo(ptr noundef %1)
  %54 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 17), align 1, !tbaa !16
  %55 = xor i8 %54, %53
  store i8 %55, ptr getelementptr inbounds nuw (i8, ptr @p, i64 17), align 1, !tbaa !16
  %56 = call fastcc i8 @foo(ptr noundef %1)
  %57 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 18), align 2, !tbaa !16
  %58 = xor i8 %57, %56
  store i8 %58, ptr getelementptr inbounds nuw (i8, ptr @p, i64 18), align 2, !tbaa !16
  %59 = call fastcc i8 @foo(ptr noundef %1)
  %60 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 19), align 1, !tbaa !16
  %61 = xor i8 %60, %59
  store i8 %61, ptr getelementptr inbounds nuw (i8, ptr @p, i64 19), align 1, !tbaa !16
  %62 = call fastcc i8 @foo(ptr noundef %1)
  %63 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 20), align 4, !tbaa !16
  %64 = xor i8 %63, %62
  store i8 %64, ptr getelementptr inbounds nuw (i8, ptr @p, i64 20), align 4, !tbaa !16
  %65 = call fastcc i8 @foo(ptr noundef %1)
  %66 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 21), align 1, !tbaa !16
  %67 = xor i8 %66, %65
  store i8 %67, ptr getelementptr inbounds nuw (i8, ptr @p, i64 21), align 1, !tbaa !16
  %68 = call fastcc i8 @foo(ptr noundef %1)
  %69 = load i8, ptr getelementptr inbounds nuw (i8, ptr @p, i64 22), align 2, !tbaa !16
  %70 = xor i8 %69, %68
  store i8 %70, ptr getelementptr inbounds nuw (i8, ptr @p, i64 22), align 2, !tbaa !16
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define internal fastcc i8 @foo(ptr noundef nonnull %0) unnamed_addr #4 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load i32, ptr %2, align 8, !tbaa !12
  %4 = add i32 %3, -1
  store i32 %4, ptr %2, align 8, !tbaa !12
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %8, label %6

6:                                                ; preds = %1
  %7 = load ptr, ptr %0, align 8, !tbaa !17
  br label %92

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %10 = load i32, ptr %9, align 4, !tbaa !6
  %11 = insertelement <4 x i32> poison, i32 %10, i64 3
  br label %12

12:                                               ; preds = %12, %8
  %13 = phi i64 [ 0, %8 ], [ %46, %12 ]
  %14 = phi <4 x i32> [ %11, %8 ], [ %20, %12 ]
  %15 = getelementptr inbounds nuw i32, ptr %9, i64 %13
  %16 = getelementptr inbounds nuw i32, ptr %9, i64 %13
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 4
  %18 = getelementptr inbounds nuw i8, ptr %16, i64 20
  %19 = load <4 x i32>, ptr %17, align 4, !tbaa !6
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !6
  %21 = shufflevector <4 x i32> %14, <4 x i32> %19, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %22 = shufflevector <4 x i32> %19, <4 x i32> %20, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %23 = and <4 x i32> %19, splat (i32 2147483646)
  %24 = and <4 x i32> %20, splat (i32 2147483646)
  %25 = and <4 x i32> %21, splat (i32 -2147483648)
  %26 = and <4 x i32> %22, splat (i32 -2147483648)
  %27 = or disjoint <4 x i32> %23, %25
  %28 = or disjoint <4 x i32> %24, %26
  %29 = lshr exact <4 x i32> %27, splat (i32 1)
  %30 = lshr exact <4 x i32> %28, splat (i32 1)
  %31 = and <4 x i32> %19, splat (i32 1)
  %32 = and <4 x i32> %20, splat (i32 1)
  %33 = icmp eq <4 x i32> %31, zeroinitializer
  %34 = icmp eq <4 x i32> %32, zeroinitializer
  %35 = select <4 x i1> %33, <4 x i32> zeroinitializer, <4 x i32> splat (i32 -1727483681)
  %36 = select <4 x i1> %34, <4 x i32> zeroinitializer, <4 x i32> splat (i32 -1727483681)
  %37 = getelementptr inbounds nuw i8, ptr %15, i64 1588
  %38 = getelementptr inbounds nuw i8, ptr %15, i64 1604
  %39 = load <4 x i32>, ptr %37, align 4, !tbaa !6
  %40 = load <4 x i32>, ptr %38, align 4, !tbaa !6
  %41 = xor <4 x i32> %35, %39
  %42 = xor <4 x i32> %36, %40
  %43 = xor <4 x i32> %41, %29
  %44 = xor <4 x i32> %42, %30
  %45 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store <4 x i32> %43, ptr %15, align 4, !tbaa !6
  store <4 x i32> %44, ptr %45, align 4, !tbaa !6
  %46 = add nuw i64 %13, 8
  %47 = icmp eq i64 %46, 224
  br i1 %47, label %48, label %12, !llvm.loop !18

48:                                               ; preds = %12
  %49 = extractelement <4 x i32> %20, i64 3
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 908
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 912
  %52 = load i32, ptr %51, align 4, !tbaa !6
  %53 = and i32 %52, 2147483646
  %54 = and i32 %49, -2147483648
  %55 = or disjoint i32 %53, %54
  %56 = lshr exact i32 %55, 1
  %57 = and i32 %52, 1
  %58 = icmp eq i32 %57, 0
  %59 = select i1 %58, i32 0, i32 -1727483681
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 2496
  %61 = load i32, ptr %60, align 4, !tbaa !6
  %62 = xor i32 %59, %61
  %63 = xor i32 %62, %56
  store i32 %63, ptr %50, align 4, !tbaa !6
  %64 = getelementptr inbounds nuw i8, ptr %0, i64 912
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 916
  %66 = load i32, ptr %65, align 4, !tbaa !6
  %67 = and i32 %66, 2147483646
  %68 = and i32 %52, -2147483648
  %69 = or disjoint i32 %67, %68
  %70 = lshr exact i32 %69, 1
  %71 = and i32 %66, 1
  %72 = icmp eq i32 %71, 0
  %73 = select i1 %72, i32 0, i32 -1727483681
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 2500
  %75 = load i32, ptr %74, align 4, !tbaa !6
  %76 = xor i32 %73, %75
  %77 = xor i32 %76, %70
  store i32 %77, ptr %64, align 4, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %0, i64 916
  %79 = getelementptr inbounds nuw i8, ptr %0, i64 920
  %80 = load i32, ptr %79, align 4, !tbaa !6
  %81 = and i32 %80, 2147483646
  %82 = and i32 %66, -2147483648
  %83 = or disjoint i32 %81, %82
  %84 = lshr exact i32 %83, 1
  %85 = and i32 %80, 1
  %86 = icmp eq i32 %85, 0
  %87 = select i1 %86, i32 0, i32 -1727483681
  %88 = getelementptr inbounds nuw i8, ptr %0, i64 2504
  %89 = load i32, ptr %88, align 4, !tbaa !6
  %90 = xor i32 %87, %89
  %91 = xor i32 %90, %84
  store i32 %91, ptr %78, align 4, !tbaa !6
  br label %92

92:                                               ; preds = %48, %6
  %93 = phi ptr [ %7, %6 ], [ %9, %48 ]
  %94 = getelementptr inbounds nuw i8, ptr %93, i64 4
  store ptr %94, ptr %0, align 8, !tbaa !17
  %95 = load i32, ptr %93, align 4, !tbaa !6
  %96 = lshr i32 %95, 11
  %97 = xor i32 %96, %95
  %98 = shl i32 %97, 7
  %99 = and i32 %98, -1658038656
  %100 = xor i32 %99, %97
  %101 = shl i32 %100, 15
  %102 = and i32 %101, 130023424
  %103 = xor i32 %102, %100
  %104 = lshr i32 %103, 19
  %105 = lshr i32 %100, 1
  %106 = xor i32 %104, %105
  %107 = trunc i32 %106 to i8
  ret i8 %107
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree noinline norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind }

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
!12 = !{!13, !7, i64 8}
!13 = !{!"S", !14, i64 0, !7, i64 8, !8, i64 12}
!14 = !{!"p1 int", !15, i64 0}
!15 = !{!"any pointer", !8, i64 0}
!16 = !{!8, !8, i64 0}
!17 = !{!13, !14, i64 0}
!18 = distinct !{!18, !11, !19, !20}
!19 = !{!"llvm.loop.isvectorized", i32 1}
!20 = !{!"llvm.loop.unroll.runtime.disable"}
