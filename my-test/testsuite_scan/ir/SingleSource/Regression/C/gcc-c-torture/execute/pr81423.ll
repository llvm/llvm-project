; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr81423.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr81423.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@ll = dso_local local_unnamed_addr global i64 0, align 8
@ull1 = dso_local local_unnamed_addr global i64 1, align 8
@ull2 = dso_local local_unnamed_addr global i64 -6438459928895745270, align 8
@ull3 = dso_local local_unnamed_addr global i64 0, align 8

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i64 0, 4294967296) i64 @foo() local_unnamed_addr #0 {
  %1 = load i64, ptr @ull1, align 8, !tbaa !6
  %2 = icmp eq i64 %1, 0
  %3 = select i1 %2, i64 2595916315, i64 2595916314
  store i64 %3, ptr @ll, align 8, !tbaa !6
  %4 = add nsw i64 %3, -2129105131
  %5 = xor i64 %4, -8165993929295883380
  %6 = add nsw i64 %5, 8165993929712309607
  %7 = shl i64 2067854353, %6
  %8 = trunc i64 %7 to i32
  %9 = load i64, ptr @ull2, align 8, !tbaa !6
  %10 = trunc i64 %9 to i32
  %11 = or i32 %10, -8651009
  %12 = add nsw i32 %11, 5
  %13 = lshr i32 %8, %12
  %14 = zext nneg i32 %13 to i64
  ret i64 %14
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i64 @foo()
  store i64 %1, ptr @ull3, align 8, !tbaa !6
  %2 = icmp eq i64 %1, 3998784
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
