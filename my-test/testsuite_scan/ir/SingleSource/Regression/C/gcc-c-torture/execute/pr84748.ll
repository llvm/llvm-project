; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr84748.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr84748.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@g0 = dso_local local_unnamed_addr global i64 0, align 8
@g1 = dso_local local_unnamed_addr global i64 0, align 8
@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i128 0, align 16
@d = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @store(i64 noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  store i64 %0, ptr @g0, align 8, !tbaa !6
  store i64 %1, ptr @g1, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #1 {
  %1 = load i32, ptr @a, align 4, !tbaa !10
  %2 = sext i32 %1 to i128
  %3 = load i128, ptr @b, align 16, !tbaa !12
  %4 = add i128 %3, %2
  store i128 %4, ptr @b, align 16, !tbaa !12
  %5 = load i32, ptr @d, align 4, !tbaa !10
  %6 = icmp ne i32 %5, 84347
  %7 = zext i1 %6 to i32
  store i32 %7, ptr @c, align 4, !tbaa !10
  %8 = trunc i128 %4 to i64
  %9 = lshr i128 %4, 64
  %10 = trunc nuw i128 %9 to i64
  store i64 %8, ptr @g0, align 8, !tbaa !6
  store i64 %10, ptr @g1, align 8, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load i32, ptr @a, align 4, !tbaa !10
  %2 = sext i32 %1 to i128
  %3 = load i128, ptr @b, align 16, !tbaa !12
  %4 = add i128 %3, %2
  store i128 %4, ptr @b, align 16, !tbaa !12
  %5 = load i32, ptr @d, align 4, !tbaa !10
  %6 = icmp ne i32 %5, 84347
  %7 = zext i1 %6 to i32
  store i32 %7, ptr @c, align 4, !tbaa !10
  %8 = trunc i128 %4 to i64
  %9 = lshr i128 %4, 64
  %10 = trunc nuw i128 %9 to i64
  store i64 %8, ptr @g0, align 8, !tbaa !6
  store i64 %10, ptr @g1, align 8, !tbaa !6
  %11 = icmp ne i64 %8, 0
  %12 = icmp ne i64 %10, 0
  %13 = select i1 %11, i1 true, i1 %12
  br i1 %13, label %14, label %15

14:                                               ; preds = %0
  tail call void @abort() #4
  unreachable

15:                                               ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"__int128", !8, i64 0}
