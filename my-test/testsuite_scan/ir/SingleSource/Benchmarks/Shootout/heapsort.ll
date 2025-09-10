; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/heapsort.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/heapsort.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@gen_random.last = internal unnamed_addr global i64 42, align 8
@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local double @gen_random(double noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr @gen_random.last, align 8, !tbaa !6
  %3 = mul nuw nsw i64 %2, 3877
  %4 = add nuw nsw i64 %3, 29573
  %5 = urem i64 %4, 139968
  store i64 %5, ptr @gen_random.last, align 8, !tbaa !6
  %6 = uitofp nneg i64 %5 to double
  %7 = fmul double %0, %6
  %8 = fdiv double %7, 1.399680e+05
  ret double %8
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @benchmark_heapsort(i32 noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #1 {
  %3 = ashr i32 %0, 1
  %4 = add nsw i32 %3, 1
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  br label %6

6:                                                ; preds = %54, %2
  %7 = phi i32 [ %0, %2 ], [ %24, %54 ]
  %8 = phi i32 [ %4, %2 ], [ %25, %54 ]
  %9 = icmp sgt i32 %8, 1
  br i1 %9, label %10, label %15

10:                                               ; preds = %6
  %11 = add nsw i32 %8, -1
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds nuw double, ptr %1, i64 %12
  %14 = load double, ptr %13, align 8, !tbaa !10
  br label %23

15:                                               ; preds = %6
  %16 = sext i32 %7 to i64
  %17 = getelementptr inbounds double, ptr %1, i64 %16
  %18 = load double, ptr %17, align 8, !tbaa !10
  %19 = load double, ptr %5, align 8, !tbaa !10
  store double %19, ptr %17, align 8, !tbaa !10
  %20 = add nsw i32 %7, -1
  %21 = icmp eq i32 %20, 1
  br i1 %21, label %22, label %23

22:                                               ; preds = %15
  store double %18, ptr %5, align 8, !tbaa !10
  ret void

23:                                               ; preds = %15, %10
  %24 = phi i32 [ %7, %10 ], [ %20, %15 ]
  %25 = phi i32 [ %11, %10 ], [ %8, %15 ]
  %26 = phi double [ %14, %10 ], [ %18, %15 ]
  %27 = shl nsw i32 %25, 1
  %28 = icmp sgt i32 %27, %24
  br i1 %28, label %54, label %29

29:                                               ; preds = %23, %49
  %30 = phi i32 [ %52, %49 ], [ %27, %23 ]
  %31 = phi i32 [ %45, %49 ], [ %25, %23 ]
  %32 = icmp slt i32 %30, %24
  %33 = sext i32 %30 to i64
  br i1 %32, label %34, label %43

34:                                               ; preds = %29
  %35 = getelementptr inbounds double, ptr %1, i64 %33
  %36 = load double, ptr %35, align 8, !tbaa !10
  %37 = or disjoint i32 %30, 1
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds double, ptr %1, i64 %38
  %40 = load double, ptr %39, align 8, !tbaa !10
  %41 = fcmp olt double %36, %40
  br i1 %41, label %42, label %43

42:                                               ; preds = %34
  br label %43

43:                                               ; preds = %29, %42, %34
  %44 = phi i64 [ %38, %42 ], [ %33, %34 ], [ %33, %29 ]
  %45 = phi i32 [ %37, %42 ], [ %30, %34 ], [ %30, %29 ]
  %46 = getelementptr inbounds double, ptr %1, i64 %44
  %47 = load double, ptr %46, align 8, !tbaa !10
  %48 = fcmp olt double %26, %47
  br i1 %48, label %49, label %54

49:                                               ; preds = %43
  %50 = sext i32 %31 to i64
  %51 = getelementptr inbounds double, ptr %1, i64 %50
  store double %47, ptr %51, align 8, !tbaa !10
  %52 = shl nsw i32 %45, 1
  %53 = icmp sgt i32 %52, %24
  br i1 %53, label %54, label %29, !llvm.loop !12

54:                                               ; preds = %43, %49, %23
  %55 = phi i32 [ %25, %23 ], [ %31, %43 ], [ %45, %49 ]
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds double, ptr %1, i64 %56
  store double %26, ptr %57, align 8, !tbaa !10
  br label %6
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #2 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !14
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #7
  %8 = trunc i64 %7 to i32
  br label %9

9:                                                ; preds = %2, %4
  %10 = phi i32 [ %8, %4 ], [ 8000000, %2 ]
  %11 = add i32 %10, 1
  %12 = sext i32 %11 to i64
  %13 = shl nsw i64 %12, 3
  %14 = tail call noalias ptr @malloc(i64 noundef %13) #8
  %15 = icmp slt i32 %10, 1
  br i1 %15, label %31, label %16

16:                                               ; preds = %9
  %17 = load i64, ptr @gen_random.last, align 8
  %18 = zext i32 %11 to i64
  br label %19

19:                                               ; preds = %16, %19
  %20 = phi i64 [ 1, %16 ], [ %28, %19 ]
  %21 = phi i64 [ %17, %16 ], [ %24, %19 ]
  %22 = mul nuw nsw i64 %21, 3877
  %23 = add nuw nsw i64 %22, 29573
  %24 = urem i64 %23, 139968
  %25 = uitofp nneg i64 %24 to double
  %26 = fdiv double %25, 1.399680e+05
  %27 = getelementptr inbounds nuw double, ptr %14, i64 %20
  store double %26, ptr %27, align 8, !tbaa !10
  %28 = add nuw nsw i64 %20, 1
  %29 = icmp eq i64 %28, %18
  br i1 %29, label %30, label %19, !llvm.loop !17

30:                                               ; preds = %19
  store i64 %24, ptr @gen_random.last, align 8, !tbaa !6
  br label %31

31:                                               ; preds = %30, %9
  %32 = ashr i32 %10, 1
  %33 = add nsw i32 %32, 1
  %34 = getelementptr inbounds nuw i8, ptr %14, i64 8
  br label %35

35:                                               ; preds = %82, %31
  %36 = phi i32 [ %10, %31 ], [ %52, %82 ]
  %37 = phi i32 [ %33, %31 ], [ %53, %82 ]
  %38 = icmp sgt i32 %37, 1
  br i1 %38, label %39, label %44

39:                                               ; preds = %35
  %40 = add nsw i32 %37, -1
  %41 = zext nneg i32 %40 to i64
  %42 = getelementptr inbounds nuw double, ptr %14, i64 %41
  %43 = load double, ptr %42, align 8, !tbaa !10
  br label %51

44:                                               ; preds = %35
  %45 = sext i32 %36 to i64
  %46 = getelementptr inbounds double, ptr %14, i64 %45
  %47 = load double, ptr %46, align 8, !tbaa !10
  %48 = load double, ptr %34, align 8, !tbaa !10
  store double %48, ptr %46, align 8, !tbaa !10
  %49 = add nsw i32 %36, -1
  %50 = icmp eq i32 %49, 1
  br i1 %50, label %86, label %51

51:                                               ; preds = %44, %39
  %52 = phi i32 [ %36, %39 ], [ %49, %44 ]
  %53 = phi i32 [ %40, %39 ], [ %37, %44 ]
  %54 = phi double [ %43, %39 ], [ %47, %44 ]
  %55 = shl nsw i32 %53, 1
  %56 = icmp sgt i32 %55, %52
  br i1 %56, label %82, label %57

57:                                               ; preds = %51, %77
  %58 = phi i32 [ %80, %77 ], [ %55, %51 ]
  %59 = phi i32 [ %73, %77 ], [ %53, %51 ]
  %60 = icmp slt i32 %58, %52
  %61 = sext i32 %58 to i64
  br i1 %60, label %62, label %71

62:                                               ; preds = %57
  %63 = getelementptr inbounds double, ptr %14, i64 %61
  %64 = load double, ptr %63, align 8, !tbaa !10
  %65 = or disjoint i32 %58, 1
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds double, ptr %14, i64 %66
  %68 = load double, ptr %67, align 8, !tbaa !10
  %69 = fcmp olt double %64, %68
  br i1 %69, label %70, label %71

70:                                               ; preds = %62
  br label %71

71:                                               ; preds = %70, %62, %57
  %72 = phi i64 [ %66, %70 ], [ %61, %62 ], [ %61, %57 ]
  %73 = phi i32 [ %65, %70 ], [ %58, %62 ], [ %58, %57 ]
  %74 = getelementptr inbounds double, ptr %14, i64 %72
  %75 = load double, ptr %74, align 8, !tbaa !10
  %76 = fcmp olt double %54, %75
  br i1 %76, label %77, label %82

77:                                               ; preds = %71
  %78 = sext i32 %59 to i64
  %79 = getelementptr inbounds double, ptr %14, i64 %78
  store double %75, ptr %79, align 8, !tbaa !10
  %80 = shl nsw i32 %73, 1
  %81 = icmp sgt i32 %80, %52
  br i1 %81, label %82, label %57, !llvm.loop !12

82:                                               ; preds = %77, %71, %51
  %83 = phi i32 [ %53, %51 ], [ %73, %77 ], [ %59, %71 ]
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds double, ptr %14, i64 %84
  store double %54, ptr %85, align 8, !tbaa !10
  br label %35

86:                                               ; preds = %44
  store double %47, ptr %34, align 8, !tbaa !10
  %87 = sext i32 %10 to i64
  %88 = getelementptr inbounds double, ptr %14, i64 %87
  %89 = load double, ptr %88, align 8, !tbaa !10
  %90 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %89)
  tail call void @free(ptr noundef nonnull %14) #7
  ret i32 0
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }
attributes #8 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 omnipotent char", !16, i64 0}
!16 = !{!"any pointer", !8, i64 0}
!17 = distinct !{!17, !13}
