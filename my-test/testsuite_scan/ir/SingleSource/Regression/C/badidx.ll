; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/badidx.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/badidx.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %6, label %4

4:                                                ; preds = %2
  %5 = tail call noalias dereferenceable_or_null(4) ptr @calloc(i64 noundef 1, i64 noundef 4) #4
  br label %15

6:                                                ; preds = %2
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !6
  %9 = tail call i64 @strtol(ptr noundef nonnull captures(none) %8, ptr noundef null, i32 noundef 10) #5
  %10 = trunc i64 %9 to i32
  %11 = shl i64 %9, 32
  %12 = ashr exact i64 %11, 32
  %13 = tail call noalias ptr @calloc(i64 noundef %12, i64 noundef 4) #4
  %14 = icmp sgt i32 %10, 0
  br i1 %14, label %15, label %49

15:                                               ; preds = %4, %6
  %16 = phi ptr [ %5, %4 ], [ %13, %6 ]
  %17 = phi i64 [ 1, %4 ], [ %12, %6 ]
  %18 = phi i64 [ 1, %4 ], [ %9, %6 ]
  %19 = and i64 %18, 4294967295
  %20 = icmp samesign ult i64 %19, 2
  br i1 %20, label %36, label %21

21:                                               ; preds = %15
  %22 = and i64 %18, 4294967294
  br label %23

23:                                               ; preds = %23, %21
  %24 = phi i64 [ 0, %21 ], [ %32, %23 ]
  %25 = or disjoint i64 %24, 1
  %26 = mul nuw nsw i64 %24, %24
  %27 = mul nuw nsw i64 %25, %25
  %28 = getelementptr inbounds nuw i32, ptr %16, i64 %24
  %29 = getelementptr inbounds nuw i32, ptr %16, i64 %25
  %30 = trunc nuw i64 %26 to i32
  %31 = trunc nuw i64 %27 to i32
  store i32 %30, ptr %28, align 4, !tbaa !11
  store i32 %31, ptr %29, align 4, !tbaa !11
  %32 = add nuw i64 %24, 2
  %33 = icmp eq i64 %32, %22
  br i1 %33, label %34, label %23, !llvm.loop !13

34:                                               ; preds = %23
  %35 = icmp eq i64 %19, %22
  br i1 %35, label %45, label %36

36:                                               ; preds = %15, %34
  %37 = phi i64 [ 0, %15 ], [ %22, %34 ]
  br label %38

38:                                               ; preds = %36, %38
  %39 = phi i64 [ %43, %38 ], [ %37, %36 ]
  %40 = mul nuw nsw i64 %39, %39
  %41 = getelementptr inbounds nuw i32, ptr %16, i64 %39
  %42 = trunc nuw i64 %40 to i32
  store i32 %42, ptr %41, align 4, !tbaa !11
  %43 = add nuw nsw i64 %39, 1
  %44 = icmp eq i64 %43, %19
  br i1 %44, label %45, label %38, !llvm.loop !17

45:                                               ; preds = %38, %34
  %46 = getelementptr i32, ptr %16, i64 %17
  %47 = getelementptr i8, ptr %46, i64 -4
  %48 = load i32, ptr %47, align 4, !tbaa !11
  br label %49

49:                                               ; preds = %45, %6
  %50 = phi i32 [ %48, %45 ], [ 0, %6 ]
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %50)
  ret i32 0
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind allocsize(0,1) }
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
!17 = distinct !{!17, !14, !15}
