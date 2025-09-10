; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/revertBits.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/revertBits.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [14 x i8] c"0x%x -> 0x%x\0A\00", align 1
@.str.3 = private unnamed_addr constant [18 x i8] c"0x%llx -> 0x%llx\0A\00", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @ReverseBits32(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i32 @llvm.bitreverse.i32(i32 %0)
  ret i32 %2
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i64 @ReverseBits64(i64 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i64 @llvm.bitreverse.i64(i64 %0)
  ret i64 %2
}

; Function Attrs: nofree nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = phi i64 [ 0, %0 ], [ %8, %1 ]
  %4 = phi i64 [ 0, %0 ], [ %10, %1 ]
  %5 = trunc nuw nsw i64 %2 to i32
  %6 = tail call i32 @ReverseBits32(i32 noundef %5)
  %7 = zext i32 %6 to i64
  %8 = add i64 %3, %7
  %9 = tail call i64 @ReverseBits64(i64 noundef %2)
  %10 = add i64 %9, %4
  %11 = add nuw nsw i64 %2, 1
  %12 = icmp eq i64 %11, 16777216
  br i1 %12, label %13, label %1, !llvm.loop !6

13:                                               ; preds = %1
  %14 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %8, i64 0
  %15 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %10, i64 0
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %36, %16 ]
  %18 = phi <2 x i64> [ <i64 0, i64 1>, %13 ], [ %37, %16 ]
  %19 = phi <2 x i64> [ %14, %13 ], [ %30, %16 ]
  %20 = phi <2 x i64> [ zeroinitializer, %13 ], [ %31, %16 ]
  %21 = phi <2 x i64> [ %15, %13 ], [ %34, %16 ]
  %22 = phi <2 x i64> [ zeroinitializer, %13 ], [ %35, %16 ]
  %23 = phi <2 x i32> [ <i32 0, i32 1>, %13 ], [ %38, %16 ]
  %24 = add <2 x i64> %18, splat (i64 2)
  %25 = add <2 x i32> %23, splat (i32 2)
  %26 = tail call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %23)
  %27 = tail call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %25)
  %28 = zext <2 x i32> %26 to <2 x i64>
  %29 = zext <2 x i32> %27 to <2 x i64>
  %30 = sub <2 x i64> %19, %28
  %31 = sub <2 x i64> %20, %29
  %32 = tail call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %18)
  %33 = tail call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %24)
  %34 = sub <2 x i64> %21, %32
  %35 = sub <2 x i64> %22, %33
  %36 = add nuw i64 %17, 4
  %37 = add <2 x i64> %18, splat (i64 4)
  %38 = add <2 x i32> %23, splat (i32 4)
  %39 = icmp eq i64 %36, 16777216
  br i1 %39, label %40, label %16, !llvm.loop !8

40:                                               ; preds = %16
  %41 = add <2 x i64> %31, %30
  %42 = tail call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %41)
  %43 = add <2 x i64> %35, %34
  %44 = tail call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %43)
  %45 = icmp ne i64 %42, 0
  %46 = icmp ne i64 %44, 0
  %47 = select i1 %45, i1 true, i1 %46
  %48 = zext i1 %47 to i32
  %49 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 305419896, i32 noundef 510274632)
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef 81985529205302085, i64 noundef -6718103380001897344)
  ret i32 %48
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.bitreverse.i32(i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.bitreverse.i64(i64) #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i32> @llvm.bitreverse.v2i32(<2 x i32>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i64> @llvm.bitreverse.v2i64(<2 x i64>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.vector.reduce.add.v2i64(<2 x i64>) #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!8 = distinct !{!8, !7, !9, !10}
!9 = !{!"llvm.loop.isvectorized", i32 1}
!10 = !{!"llvm.loop.unroll.runtime.disable"}
