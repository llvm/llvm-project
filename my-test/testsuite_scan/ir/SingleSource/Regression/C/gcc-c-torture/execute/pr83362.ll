; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr83362.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr83362.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i8 0, align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call fastcc i32 @foo()
  %2 = icmp eq i32 %1, 253
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal fastcc range(i32 0, 88030) i32 @foo() unnamed_addr #1 {
  %1 = load i32, ptr @b, align 4, !tbaa !6
  %2 = load i32, ptr @a, align 4, !tbaa !6
  %3 = freeze i32 %1
  %4 = freeze i32 %2
  %5 = icmp ugt i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %0
  %7 = load i32, ptr @e, align 4, !tbaa !6
  br label %8

8:                                                ; preds = %6, %8
  %9 = phi i32 [ %11, %8 ], [ %7, %6 ]
  %10 = udiv i32 %9, 4095
  %11 = add nsw i32 %10, -3
  %12 = icmp ugt i32 %11, 88029
  br i1 %12, label %8, label %13, !llvm.loop !10

13:                                               ; preds = %8, %0
  %14 = phi i32 [ 0, %0 ], [ %11, %8 ]
  store i32 %14, ptr @e, align 4, !tbaa !6
  store i32 0, ptr @d, align 4, !tbaa !6
  ret i32 %14
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
