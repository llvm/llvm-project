; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/ary3.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/ary3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [7 x i8] c"%d %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #5
  %8 = trunc i64 %7 to i32
  br label %9

9:                                                ; preds = %2, %4
  %10 = phi i32 [ %8, %4 ], [ 1500000, %2 ]
  %11 = sext i32 %10 to i64
  %12 = tail call noalias ptr @calloc(i64 noundef %11, i64 noundef 4) #6
  %13 = tail call noalias ptr @calloc(i64 noundef %11, i64 noundef 4) #6
  %14 = icmp sgt i32 %10, 0
  br i1 %14, label %15, label %88

15:                                               ; preds = %9
  %16 = zext nneg i32 %10 to i64
  %17 = icmp ult i32 %10, 8
  br i1 %17, label %34, label %18

18:                                               ; preds = %15
  %19 = and i64 %16, 2147483640
  br label %20

20:                                               ; preds = %20, %18
  %21 = phi i64 [ 0, %18 ], [ %29, %20 ]
  %22 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %18 ], [ %30, %20 ]
  %23 = getelementptr inbounds nuw i32, ptr %12, i64 %21
  %24 = trunc <4 x i64> %22 to <4 x i32>
  %25 = add <4 x i32> %24, splat (i32 1)
  %26 = trunc <4 x i64> %22 to <4 x i32>
  %27 = add <4 x i32> %26, splat (i32 5)
  %28 = getelementptr inbounds nuw i8, ptr %23, i64 16
  store <4 x i32> %25, ptr %23, align 4, !tbaa !11
  store <4 x i32> %27, ptr %28, align 4, !tbaa !11
  %29 = add nuw i64 %21, 8
  %30 = add <4 x i64> %22, splat (i64 8)
  %31 = icmp eq i64 %29, %19
  br i1 %31, label %32, label %20, !llvm.loop !13

32:                                               ; preds = %20
  %33 = icmp eq i64 %19, %16
  br i1 %33, label %36, label %34

34:                                               ; preds = %15, %32
  %35 = phi i64 [ 0, %15 ], [ %19, %32 ]
  br label %77

36:                                               ; preds = %77, %32
  %37 = zext nneg i32 %10 to i64
  %38 = icmp ult i32 %10, 8
  %39 = and i64 %16, 2147483640
  %40 = sub nsw i64 %37, %39
  %41 = icmp eq i64 %39, %16
  br label %42

42:                                               ; preds = %36, %74
  %43 = phi i32 [ %75, %74 ], [ 0, %36 ]
  br i1 %38, label %63, label %44

44:                                               ; preds = %42, %44
  %45 = phi i64 [ %60, %44 ], [ 0, %42 ]
  %46 = xor i64 %45, -1
  %47 = add i64 %46, %37
  %48 = getelementptr inbounds nuw i32, ptr %12, i64 %47
  %49 = getelementptr inbounds i8, ptr %48, i64 -12
  %50 = getelementptr inbounds i8, ptr %48, i64 -28
  %51 = load <4 x i32>, ptr %49, align 4, !tbaa !11
  %52 = load <4 x i32>, ptr %50, align 4, !tbaa !11
  %53 = getelementptr inbounds nuw i32, ptr %13, i64 %47
  %54 = getelementptr inbounds i8, ptr %53, i64 -12
  %55 = getelementptr inbounds i8, ptr %53, i64 -28
  %56 = load <4 x i32>, ptr %54, align 4, !tbaa !11
  %57 = load <4 x i32>, ptr %55, align 4, !tbaa !11
  %58 = add nsw <4 x i32> %56, %51
  %59 = add nsw <4 x i32> %57, %52
  store <4 x i32> %58, ptr %54, align 4, !tbaa !11
  store <4 x i32> %59, ptr %55, align 4, !tbaa !11
  %60 = add nuw i64 %45, 8
  %61 = icmp eq i64 %60, %39
  br i1 %61, label %62, label %44, !llvm.loop !17

62:                                               ; preds = %44
  br i1 %41, label %74, label %63

63:                                               ; preds = %42, %62
  %64 = phi i64 [ %37, %42 ], [ %40, %62 ]
  br label %65

65:                                               ; preds = %63, %65
  %66 = phi i64 [ %67, %65 ], [ %64, %63 ]
  %67 = add nsw i64 %66, -1
  %68 = getelementptr inbounds nuw i32, ptr %12, i64 %67
  %69 = load i32, ptr %68, align 4, !tbaa !11
  %70 = getelementptr inbounds nuw i32, ptr %13, i64 %67
  %71 = load i32, ptr %70, align 4, !tbaa !11
  %72 = add nsw i32 %71, %69
  store i32 %72, ptr %70, align 4, !tbaa !11
  %73 = icmp sgt i64 %66, 1
  br i1 %73, label %65, label %74, !llvm.loop !18

74:                                               ; preds = %65, %62
  %75 = add nuw nsw i32 %43, 1
  %76 = icmp eq i32 %75, 1000
  br i1 %76, label %83, label %42, !llvm.loop !19

77:                                               ; preds = %34, %77
  %78 = phi i64 [ %79, %77 ], [ %35, %34 ]
  %79 = add nuw nsw i64 %78, 1
  %80 = getelementptr inbounds nuw i32, ptr %12, i64 %78
  %81 = trunc nuw nsw i64 %79 to i32
  store i32 %81, ptr %80, align 4, !tbaa !11
  %82 = icmp eq i64 %79, %16
  br i1 %82, label %36, label %77, !llvm.loop !20

83:                                               ; preds = %74
  %84 = load i32, ptr %13, align 4, !tbaa !11
  %85 = getelementptr i32, ptr %13, i64 %11
  %86 = getelementptr i8, ptr %85, i64 -4
  %87 = load i32, ptr %86, align 4, !tbaa !11
  br label %88

88:                                               ; preds = %9, %83
  %89 = phi i32 [ %87, %83 ], [ 0, %9 ]
  %90 = phi i32 [ %84, %83 ], [ 0, %9 ]
  %91 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %90, i32 noundef %89)
  tail call void @free(ptr noundef %12) #5
  tail call void @free(ptr noundef %13) #5
  ret i32 0
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #4

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }
attributes #6 = { nounwind allocsize(0,1) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !15, !16}
!18 = distinct !{!18, !14, !16, !15}
!19 = distinct !{!19, !14}
!20 = distinct !{!20, !14, !16, !15}
