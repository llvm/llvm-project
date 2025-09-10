; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20070614-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20070614-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v = dso_local local_unnamed_addr global { double, double } { double 3.000000e+00, double 1.000000e+00 }, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo([2 x double] noundef alignstack(8) %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = extractvalue [2 x double] %0, 0
  %4 = extractvalue [2 x double] %0, 1
  %5 = load double, ptr @v, align 8
  %6 = load double, ptr getelementptr inbounds nuw (i8, ptr @v, i64 8), align 8
  %7 = fcmp une double %3, %5
  %8 = fcmp une double %4, %6
  %9 = or i1 %7, %8
  br i1 %9, label %10, label %11

10:                                               ; preds = %2
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local { double, double } @bar([2 x double] noundef alignstack(8) %0) local_unnamed_addr #2 {
  %2 = load double, ptr @v, align 8
  %3 = load double, ptr getelementptr inbounds nuw (i8, ptr @v, i64 8), align 8
  %4 = insertvalue { double, double } poison, double %2, 0
  %5 = insertvalue { double, double } %4, double %3, 1
  ret { double, double } %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @baz() local_unnamed_addr #0 {
  %1 = load double, ptr @v, align 8
  %2 = load double, ptr getelementptr inbounds nuw (i8, ptr @v, i64 8), align 8
  %3 = fcmp uno double %1, %2
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  ret i32 0

5:                                                ; preds = %0
  tail call void @abort() #3
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load double, ptr @v, align 8
  %2 = load double, ptr getelementptr inbounds nuw (i8, ptr @v, i64 8), align 8
  %3 = fcmp uno double %1, %2
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %0
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
