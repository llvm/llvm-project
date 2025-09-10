; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr17133.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr17133.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@foo = dso_local local_unnamed_addr global i32 0, align 4
@bar = dso_local local_unnamed_addr global ptr null, align 8
@baz = dso_local local_unnamed_addr global i32 100, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local ptr @pure_alloc() local_unnamed_addr #0 {
  %1 = load ptr, ptr @bar, align 8, !tbaa !6
  %2 = load i32, ptr @baz, align 4, !tbaa !10
  %3 = load i32, ptr @foo, align 4, !tbaa !10
  %4 = add nsw i32 %3, 2
  %5 = icmp ult i32 %4, %2
  br i1 %5, label %9, label %6

6:                                                ; preds = %0
  %7 = icmp ugt i32 %2, 2
  br label %8

8:                                                ; preds = %8, %6
  br i1 %7, label %9, label %8, !llvm.loop !12

9:                                                ; preds = %8, %0
  %10 = phi i32 [ %3, %0 ], [ 0, %8 ]
  %11 = phi i32 [ %4, %0 ], [ 2, %8 ]
  store i32 %11, ptr @foo, align 4, !tbaa !10
  %12 = sext i32 %10 to i64
  %13 = getelementptr inbounds i8, ptr %1, i64 %12
  %14 = ptrtoint ptr %13 to i64
  %15 = and i64 %14, 4294967294
  %16 = inttoptr i64 %15 to ptr
  ret ptr %16
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @baz, align 4, !tbaa !10
  %2 = load i32, ptr @foo, align 4, !tbaa !10
  %3 = add nsw i32 %2, 2
  %4 = icmp ult i32 %3, %1
  br i1 %4, label %9, label %5

5:                                                ; preds = %0
  %6 = icmp ugt i32 %1, 2
  br i1 %6, label %8, label %7, !llvm.loop !12

7:                                                ; preds = %5, %7
  br label %7

8:                                                ; preds = %5
  store i32 2, ptr @foo, align 4, !tbaa !10
  br label %12

9:                                                ; preds = %0
  store i32 %3, ptr @foo, align 4, !tbaa !10
  %10 = icmp eq i32 %3, 0
  br i1 %10, label %11, label %12

11:                                               ; preds = %9
  tail call void @abort() #3
  unreachable

12:                                               ; preds = %8, %9
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.peeled.count", i32 1}
