; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr70460.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr70460.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@foo.b = internal unnamed_addr constant [2 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr blockaddress(@foo, %7) to i64), i64 ptrtoint (ptr blockaddress(@foo, %13) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr blockaddress(@foo, %10) to i64), i64 ptrtoint (ptr blockaddress(@foo, %13) to i64)) to i32)], align 4
@c = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = sext i32 %0 to i64
  %3 = getelementptr inbounds i32, ptr @foo.b, i64 %2
  %4 = load i32, ptr %3, align 4, !tbaa !6
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds i8, ptr blockaddress(@foo, %13), i64 %5
  indirectbr ptr %6, [label %7, label %13, label %10]

7:                                                ; preds = %1
  %8 = load i32, ptr @c, align 4, !tbaa !6
  %9 = add nsw i32 %8, 2
  store i32 %9, ptr @c, align 4, !tbaa !6
  br label %10

10:                                               ; preds = %7, %1
  %11 = load i32, ptr @c, align 4, !tbaa !6
  %12 = add nsw i32 %11, 1
  store i32 %12, ptr @c, align 4, !tbaa !6
  br label %13

13:                                               ; preds = %10, %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @foo(i32 noundef 0)
  %1 = load i32, ptr @c, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 3
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

4:                                                ; preds = %0
  tail call void @foo(i32 noundef 1)
  %5 = load i32, ptr @c, align 4, !tbaa !6
  %6 = icmp eq i32 %5, 4
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %4
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
