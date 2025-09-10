; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr81555.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr81555.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 1, align 4
@d = dso_local local_unnamed_addr global i32 4014, align 4
@e = dso_local local_unnamed_addr global i32 58230, align 4
@b = dso_local local_unnamed_addr global i8 0, align 4
@f = dso_local local_unnamed_addr global i8 1, align 4
@g = dso_local local_unnamed_addr global i8 1, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = load i8, ptr @b, align 4, !tbaa !10, !range !12, !noundef !13
  %3 = zext nneg i8 %2 to i32
  %4 = icmp eq i32 %1, %3
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  store i8 0, ptr @f, align 4, !tbaa !10
  br label %6

6:                                                ; preds = %5, %0
  %7 = load i32, ptr @e, align 4, !tbaa !6
  %8 = and i32 %7, 1
  %9 = select i1 %4, i32 0, i32 %8
  %10 = load i32, ptr @d, align 4, !tbaa !6
  %11 = and i32 %9, %10
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %14, label %13

13:                                               ; preds = %6
  store i8 0, ptr @g, align 4, !tbaa !14
  br label %14

14:                                               ; preds = %13, %6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = load i8, ptr @b, align 4, !tbaa !10, !range !12, !noundef !13
  %3 = zext nneg i8 %2 to i32
  %4 = icmp eq i32 %1, %3
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  store i8 0, ptr @f, align 4, !tbaa !10
  br label %6

6:                                                ; preds = %5, %0
  %7 = load i32, ptr @e, align 4, !tbaa !6
  %8 = and i32 %7, 1
  %9 = select i1 %4, i32 0, i32 %8
  %10 = load i32, ptr @d, align 4, !tbaa !6
  %11 = and i32 %9, %10
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %14, label %13

13:                                               ; preds = %6
  store i8 0, ptr @g, align 4, !tbaa !14
  br label %20

14:                                               ; preds = %6
  %15 = load i8, ptr @g, align 4
  %16 = icmp ne i8 %15, 1
  %17 = load i8, ptr @f, align 4, !tbaa !10, !range !12, !noundef !13
  %18 = trunc nuw i8 %17 to i1
  %19 = select i1 %18, i1 true, i1 %16
  br i1 %19, label %20, label %21

20:                                               ; preds = %13, %14
  tail call void @abort() #3
  unreachable

21:                                               ; preds = %14
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!10 = !{!11, !11, i64 0}
!11 = !{!"_Bool", !8, i64 0}
!12 = !{i8 0, i8 2}
!13 = !{}
!14 = !{!8, !8, i64 0}
