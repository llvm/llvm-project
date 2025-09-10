; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/inf-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/inf-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local void @test(double noundef %0, double noundef %1) local_unnamed_addr #0 {
  %3 = fcmp oeq double %0, 0x7FF0000000000000
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %2
  %6 = fcmp oeq double %0, 0xFFF0000000000000
  br i1 %6, label %7, label %8

7:                                                ; preds = %5
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %5
  %9 = fcmp oeq double %1, 0xFFF0000000000000
  br i1 %9, label %10, label %11

10:                                               ; preds = %8
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %8
  %12 = fcmp une double %1, 0x7FF0000000000000
  br i1 %12, label %13, label %14

13:                                               ; preds = %11
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %11
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @testf(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = fcmp oeq float %0, 0x7FF0000000000000
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %2
  %6 = fcmp oeq float %0, 0xFFF0000000000000
  br i1 %6, label %7, label %8

7:                                                ; preds = %5
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %5
  %9 = fcmp oeq float %1, 0xFFF0000000000000
  br i1 %9, label %10, label %11

10:                                               ; preds = %8
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %8
  %12 = fcmp une float %1, 0x7FF0000000000000
  br i1 %12, label %13, label %14

13:                                               ; preds = %11
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %11
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testl(fp128 noundef %0, fp128 noundef %1) local_unnamed_addr #0 {
  %3 = fcmp oeq fp128 %0, 0xL00000000000000007FFF000000000000
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %2
  %6 = fcmp oeq fp128 %0, 0xL0000000000000000FFFF000000000000
  br i1 %6, label %7, label %8

7:                                                ; preds = %5
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %5
  %9 = fcmp oeq fp128 %1, 0xL0000000000000000FFFF000000000000
  br i1 %9, label %10, label %11

10:                                               ; preds = %8
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %8
  %12 = fcmp une fp128 %1, 0xL00000000000000007FFF000000000000
  br i1 %12, label %13, label %14

13:                                               ; preds = %11
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
