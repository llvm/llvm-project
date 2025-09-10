; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Perm.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Perm.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.element = type { i32, i32 }
%struct.complex = type { float, float }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@permarray = dso_local local_unnamed_addr global [11 x i32] zeroinitializer, align 4
@pctr = dso_local local_unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@value = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@fixed = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@floated = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@tree = dso_local local_unnamed_addr global ptr null, align 8
@stack = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@cellspace = dso_local local_unnamed_addr global [19 x %struct.element] zeroinitializer, align 4
@freelist = dso_local local_unnamed_addr global i32 0, align 4
@movesdone = dso_local local_unnamed_addr global i32 0, align 4
@ima = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imb = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imr = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@rma = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmb = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmr = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@piececount = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@class = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@piecemax = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@puzzl = dso_local local_unnamed_addr global [512 x i32] zeroinitializer, align 4
@p = dso_local local_unnamed_addr global [13 x [512 x i32]] zeroinitializer, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@kount = dso_local local_unnamed_addr global i32 0, align 4
@sortlist = dso_local local_unnamed_addr global [5001 x i32] zeroinitializer, align 4
@biggest = dso_local local_unnamed_addr global i32 0, align 4
@littlest = dso_local local_unnamed_addr global i32 0, align 4
@top = dso_local local_unnamed_addr global i32 0, align 4
@z = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@w = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@e = dso_local local_unnamed_addr global [130 x %struct.complex] zeroinitializer, align 4
@zr = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@zi = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@str = private unnamed_addr constant [16 x i8] c" Error in Perm.\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Initrand() local_unnamed_addr #0 {
  store i64 74755, ptr @seed, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 65536) i32 @Rand() local_unnamed_addr #1 {
  %1 = load i64, ptr @seed, align 8, !tbaa !6
  %2 = mul nsw i64 %1, 1309
  %3 = add nsw i64 %2, 13849
  %4 = and i64 %3, 65535
  store i64 %4, ptr @seed, align 8, !tbaa !6
  %5 = trunc nuw nsw i64 %4 to i32
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @Swap(ptr noundef captures(none) %0, ptr noundef captures(none) %1) local_unnamed_addr #2 {
  %3 = load i32, ptr %0, align 4, !tbaa !10
  %4 = load i32, ptr %1, align 4, !tbaa !10
  store i32 %4, ptr %0, align 4, !tbaa !10
  store i32 %3, ptr %1, align 4, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Initialize() local_unnamed_addr #0 {
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 4), align 4, !tbaa !10
  store <2 x i32> <i32 4, i32 5>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 20), align 4, !tbaa !10
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 28), align 4, !tbaa !10
  ret void
}

; Function Attrs: nofree nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Permute(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @pctr, align 4, !tbaa !10
  %3 = add i32 %2, 1
  store i32 %3, ptr @pctr, align 4, !tbaa !10
  %4 = icmp eq i32 %0, 1
  br i1 %4, label %21, label %5

5:                                                ; preds = %1
  %6 = add nsw i32 %0, -1
  tail call void @Permute(i32 noundef %6)
  %7 = icmp sgt i32 %0, 1
  br i1 %7, label %8, label %21

8:                                                ; preds = %5
  %9 = zext nneg i32 %0 to i64
  %10 = getelementptr inbounds nuw i32, ptr @permarray, i64 %9
  %11 = zext nneg i32 %6 to i64
  br label %12

12:                                               ; preds = %8, %12
  %13 = phi i64 [ %11, %8 ], [ %19, %12 ]
  %14 = getelementptr inbounds nuw i32, ptr @permarray, i64 %13
  %15 = load i32, ptr %10, align 4, !tbaa !10
  %16 = load i32, ptr %14, align 4, !tbaa !10
  store i32 %16, ptr %10, align 4, !tbaa !10
  store i32 %15, ptr %14, align 4, !tbaa !10
  tail call void @Permute(i32 noundef %6)
  %17 = load i32, ptr %10, align 4, !tbaa !10
  %18 = load i32, ptr %14, align 4, !tbaa !10
  store i32 %18, ptr %10, align 4, !tbaa !10
  store i32 %17, ptr %14, align 4, !tbaa !10
  %19 = add nsw i64 %13, -1
  %20 = icmp samesign ugt i64 %13, 1
  br i1 %20, label %12, label %21, !llvm.loop !12

21:                                               ; preds = %12, %5, %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Perm() local_unnamed_addr #4 {
  store i32 0, ptr @pctr, align 4, !tbaa !10
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 4), align 4, !tbaa !10
  store <2 x i32> <i32 4, i32 5>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 20), align 4, !tbaa !10
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 28), align 4, !tbaa !10
  tail call void @Permute(i32 noundef 7)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 4), align 4, !tbaa !10
  store <2 x i32> <i32 4, i32 5>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 20), align 4, !tbaa !10
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 28), align 4, !tbaa !10
  tail call void @Permute(i32 noundef 7)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 4), align 4, !tbaa !10
  store <2 x i32> <i32 4, i32 5>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 20), align 4, !tbaa !10
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 28), align 4, !tbaa !10
  tail call void @Permute(i32 noundef 7)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 4), align 4, !tbaa !10
  store <2 x i32> <i32 4, i32 5>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 20), align 4, !tbaa !10
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 28), align 4, !tbaa !10
  tail call void @Permute(i32 noundef 7)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 4), align 4, !tbaa !10
  store <2 x i32> <i32 4, i32 5>, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 20), align 4, !tbaa !10
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @permarray, i64 28), align 4, !tbaa !10
  tail call void @Permute(i32 noundef 7)
  %1 = load i32, ptr @pctr, align 4, !tbaa !10
  %2 = icmp eq i32 %1, 43300
  br i1 %2, label %6, label %3

3:                                                ; preds = %0
  %4 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %5 = load i32, ptr @pctr, align 4, !tbaa !10
  br label %6

6:                                                ; preds = %3, %0
  %7 = phi i32 [ %5, %3 ], [ 43300, %0 ]
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %7)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #5

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  tail call void @Perm()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind }

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
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
