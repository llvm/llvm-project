; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr34415.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr34415.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"Bbb:\00", align 1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local ptr @foo(ptr noundef readonly captures(ret: address, provenance) %0) local_unnamed_addr #0 {
  br label %2

2:                                                ; preds = %17, %1
  %3 = phi ptr [ %0, %1 ], [ %19, %17 ]
  %4 = phi ptr [ undef, %1 ], [ %3, %17 ]
  %5 = phi i32 [ 1, %1 ], [ %20, %17 ]
  %6 = load i8, ptr %3, align 1, !tbaa !6
  %7 = zext i8 %6 to i32
  %8 = add i8 %6, -97
  %9 = icmp ult i8 %8, 26
  %10 = add nsw i32 %7, -32
  %11 = select i1 %9, i32 %10, i32 %7
  switch i32 %11, label %21 [
    i32 66, label %17
    i32 65, label %12
  ]

12:                                               ; preds = %2, %12
  %13 = phi ptr [ %14, %12 ], [ %3, %2 ]
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 1
  %15 = load i8, ptr %14, align 1, !tbaa !6
  %16 = icmp eq i8 %15, 43
  br i1 %16, label %12, label %17, !llvm.loop !9

17:                                               ; preds = %12, %2
  %18 = phi ptr [ %3, %2 ], [ %14, %12 ]
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 1
  %20 = add nuw nsw i32 %5, 1
  br label %2

21:                                               ; preds = %2
  %22 = icmp samesign ugt i32 %5, 2
  %23 = icmp eq i8 %6, 58
  %24 = and i1 %22, %23
  %25 = select i1 %24, ptr %4, ptr %3
  ret ptr %25
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #1 {
  %1 = tail call ptr @foo(ptr noundef nonnull @.str)
  %2 = icmp ne ptr %1, getelementptr inbounds nuw (i8, ptr @.str, i64 2)
  %3 = zext i1 %2 to i32
  ret i32 %3
}

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
