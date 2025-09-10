; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20051110-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20051110-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@bytes = dso_local local_unnamed_addr global [5 x i8] zeroinitializer, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @add_unwind_adjustsp(i64 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i64 %0, -516
  %3 = icmp ult i64 %2, 4
  br i1 %3, label %17, label %4

4:                                                ; preds = %1
  %5 = ashr i64 %2, 2
  br label %6

6:                                                ; preds = %4, %6
  %7 = phi i64 [ 0, %4 ], [ %16, %6 ]
  %8 = phi i64 [ %5, %4 ], [ %12, %6 ]
  %9 = trunc i64 %8 to i8
  %10 = and i8 %9, 127
  %11 = getelementptr inbounds nuw i8, ptr @bytes, i64 %7
  %12 = lshr i64 %8, 7
  %13 = icmp ult i64 %8, 128
  %14 = select i1 %13, i8 0, i8 -128
  %15 = or disjoint i8 %10, %14
  store i8 %15, ptr %11, align 1, !tbaa !6
  %16 = add nuw nsw i64 %7, 1
  br i1 %13, label %17, label %6, !llvm.loop !9

17:                                               ; preds = %6, %1
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i8 -120, ptr @bytes, align 4, !tbaa !6
  store i8 7, ptr getelementptr inbounds nuw (i8, ptr @bytes, i64 1), align 1, !tbaa !6
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
