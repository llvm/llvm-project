; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/conditional-gnu-ext-cxx.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/conditional-gnu-ext-cxx.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@_ZZ10getComplexCiE5count = internal unnamed_addr global i32 0, align 4
@global = dso_local global i32 1, align 4
@_ZZ4condvE5count = internal unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local noundef i64 @_Z10getComplexCi(i64 noundef returned %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @_ZZ10getComplexCiE5count, align 4, !tbaa !6
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr @_ZZ10getComplexCiE5count, align 4, !tbaa !6
  %4 = icmp eq i32 %2, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @abort()
  unreachable

6:                                                ; preds = %1
  ret i64 %0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local noundef i64 @_Z10cmplx_testv() local_unnamed_addr #0 {
  %1 = load i32, ptr @_ZZ10getComplexCiE5count, align 4, !tbaa !6
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @_ZZ10getComplexCiE5count, align 4, !tbaa !6
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort()
  unreachable

5:                                                ; preds = %0
  ret i64 8589934593
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @_Z3fooRi(ptr noundef nonnull align 4 captures(none) dereferenceable(4) %0) local_unnamed_addr #2 {
  %2 = load i32, ptr %0, align 4, !tbaa !6
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr %0, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local noundef nonnull align 4 dereferenceable(4) ptr @_Z4condv() local_unnamed_addr #0 {
  %1 = load i32, ptr @_ZZ4condvE5count, align 4, !tbaa !6
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @_ZZ4condvE5count, align 4, !tbaa !6
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort()
  unreachable

5:                                                ; preds = %0
  ret ptr @global
}

; Function Attrs: mustprogress nofree norecurse nounwind uwtable
define dso_local noundef range(i32 -2147483648, 2147483646) i32 @main() local_unnamed_addr #3 {
  %1 = load i32, ptr @_ZZ10getComplexCiE5count, align 4, !tbaa !6
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @_ZZ10getComplexCiE5count, align 4, !tbaa !6
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort()
  unreachable

5:                                                ; preds = %0
  %6 = load i32, ptr @_ZZ4condvE5count, align 4, !tbaa !6
  %7 = add nsw i32 %6, 1
  store i32 %7, ptr @_ZZ4condvE5count, align 4, !tbaa !6
  %8 = icmp eq i32 %6, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %5
  tail call void @abort()
  unreachable

10:                                               ; preds = %5
  %11 = load i32, ptr @global, align 4, !tbaa !6
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %16, label %13

13:                                               ; preds = %10
  %14 = add nsw i32 %11, 1
  store i32 %14, ptr @global, align 4, !tbaa !6
  %15 = add nsw i32 %11, -1
  br label %16

16:                                               ; preds = %10, %13
  %17 = phi i32 [ -2, %10 ], [ %15, %13 ]
  ret i32 %17
}

attributes #0 = { mustprogress nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!9 = !{!"Simple C++ TBAA"}
