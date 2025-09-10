; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20060905-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20060905-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@s = dso_local global [256 x [3 x i8]] zeroinitializer, align 1
@g = dso_local local_unnamed_addr global i8 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %10, %0
  %2 = phi i64 [ 0, %0 ], [ %12, %10 ]
  %3 = phi i32 [ 0, %0 ], [ %11, %10 ]
  %4 = icmp samesign ugt i64 %2, 127
  br i1 %4, label %5, label %10

5:                                                ; preds = %1
  %6 = getelementptr [3 x i8], ptr @s, i64 %2
  %7 = getelementptr i8, ptr %6, i64 -384
  %8 = load volatile i8, ptr %7, align 1, !tbaa !6
  store i8 %8, ptr @g, align 4, !tbaa !6
  %9 = add nsw i32 %3, 1
  br label %10

10:                                               ; preds = %5, %1
  %11 = phi i32 [ %9, %5 ], [ %3, %1 ]
  %12 = add nuw nsw i64 %2, 1
  %13 = icmp eq i64 %12, 256
  br i1 %13, label %14, label %1, !llvm.loop !9

14:                                               ; preds = %10
  %15 = icmp eq i32 %11, 128
  br i1 %15, label %17, label %16

16:                                               ; preds = %14
  tail call void @abort() #2
  unreachable

17:                                               ; preds = %14
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

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
