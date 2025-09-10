; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33669.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33669.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local i64 @foo(ptr noundef captures(none) %0, i64 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = load i32, ptr %0, align 4, !tbaa !6
  %5 = freeze i32 %4
  %6 = zext i32 %5 to i64
  %7 = srem i64 %1, %6
  %8 = sub nsw i64 %1, %7
  %9 = trunc i64 %7 to i32
  %10 = add i32 %2, -1
  %11 = add i32 %10, %5
  %12 = add i32 %11, %9
  %13 = urem i32 %12, %5
  %14 = sub nuw i32 %12, %13
  %15 = icmp ult i32 %5, %14
  br i1 %15, label %21, label %16

16:                                               ; preds = %3
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %18 = load i32, ptr %17, align 4, !tbaa !11
  %19 = icmp ugt i32 %18, %5
  br i1 %19, label %20, label %21

20:                                               ; preds = %16
  store i32 %5, ptr %17, align 4, !tbaa !11
  br label %21

21:                                               ; preds = %16, %20, %3
  %22 = phi i64 [ -1, %3 ], [ %8, %20 ], [ %8, %16 ]
  ret i64 %22
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"foo_t", !8, i64 0, !8, i64 4}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 4}
