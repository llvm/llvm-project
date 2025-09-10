; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/compare-fp-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ieee/compare-fp-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@pinf = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@ninf = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@NaN = dso_local local_unnamed_addr global float 0.000000e+00, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @iuneq(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp ueq float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  ret i32 undef
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @ieq(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp oeq float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @iltgt(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp one float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  %9 = fcmp uno float %0, %1
  br i1 %9, label %15, label %10

10:                                               ; preds = %8
  %11 = fcmp olt float %0, %1
  br i1 %11, label %15, label %12

12:                                               ; preds = %10
  %13 = fcmp ogt float %0, %1
  %14 = zext i1 %13 to i32
  br label %15

15:                                               ; preds = %10, %12, %8
  %16 = phi i32 [ 0, %8 ], [ 1, %10 ], [ %14, %12 ]
  %17 = icmp eq i32 %16, %2
  br i1 %17, label %19, label %18

18:                                               ; preds = %15
  tail call void @abort() #3
  unreachable

19:                                               ; preds = %15
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @ine(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp une float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @iunlt(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp ult float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @ilt(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp uno float %0, %1
  %5 = fcmp olt float %0, %1
  %6 = zext i1 %5 to i32
  %7 = select i1 %4, i32 0, i32 %6
  %8 = icmp eq i32 %7, %2
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %3
  %11 = icmp eq i32 %2, %6
  br i1 %11, label %13, label %12

12:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %10
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @iunle(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp ule float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @ile(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp uno float %0, %1
  %5 = fcmp ole float %0, %1
  %6 = zext i1 %5 to i32
  %7 = select i1 %4, i32 0, i32 %6
  %8 = icmp eq i32 %7, %2
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %3
  %11 = icmp eq i32 %2, %6
  br i1 %11, label %13, label %12

12:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %10
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @iungt(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp ugt float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @igt(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp uno float %0, %1
  %5 = fcmp ogt float %0, %1
  %6 = zext i1 %5 to i32
  %7 = select i1 %4, i32 0, i32 %6
  %8 = icmp eq i32 %7, %2
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %3
  %11 = icmp eq i32 %2, %6
  br i1 %11, label %13, label %12

12:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %10
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @iunge(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp uge float %0, %1
  %5 = zext i1 %4 to i32
  %6 = icmp eq i32 %2, %5
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %3
  ret i32 undef
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @ige(float noundef %0, float noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = fcmp uno float %0, %1
  %5 = fcmp oge float %0, %1
  %6 = zext i1 %5 to i32
  %7 = select i1 %4, i32 0, i32 %6
  %8 = icmp eq i32 %7, %2
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %3
  %11 = icmp eq i32 %2, %6
  br i1 %11, label %13, label %12

12:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %10
  ret i32 undef
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store float 0x7FF0000000000000, ptr @pinf, align 4, !tbaa !6
  store float 0xFFF0000000000000, ptr @ninf, align 4, !tbaa !6
  store float 0x7FF8000000000000, ptr @NaN, align 4, !tbaa !6
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
