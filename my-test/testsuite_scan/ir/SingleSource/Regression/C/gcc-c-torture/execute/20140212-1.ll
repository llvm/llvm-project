; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20140212-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20140212-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@d = dso_local local_unnamed_addr global i32 1, align 4
@f = dso_local local_unnamed_addr global i32 1, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@k = dso_local local_unnamed_addr global i32 0, align 4
@j = dso_local local_unnamed_addr global i8 0, align 4
@g = dso_local local_unnamed_addr global i8 0, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1() local_unnamed_addr #0 {
  store i32 0, ptr @e, align 4, !tbaa !6
  store i32 0, ptr @c, align 4, !tbaa !6
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp ne i32 %1, 0
  %3 = load i32, ptr @b, align 4
  %4 = icmp ne i32 %3, 0
  %5 = select i1 %2, i1 %4, i1 false
  %6 = zext i1 %5 to i32
  store i32 %6, ptr @k, align 4, !tbaa !6
  %7 = select i1 %5, i8 54, i8 0
  store i8 %7, ptr @j, align 4, !tbaa !10
  %8 = mul i8 %7, -109
  store i8 %8, ptr @g, align 4, !tbaa !10
  %9 = load i32, ptr @d, align 4, !tbaa !6
  %10 = icmp eq i32 %9, 0
  %11 = load i32, ptr @f, align 4, !tbaa !6
  %12 = icmp eq i32 %11, 0
  br i1 %10, label %13, label %15

13:                                               ; preds = %0
  store i32 9, ptr @i, align 4, !tbaa !6
  store i32 9, ptr @h, align 4, !tbaa !6
  br i1 %12, label %14, label %17

14:                                               ; preds = %13, %14
  br label %14

15:                                               ; preds = %0
  store i32 1, ptr @c, align 4, !tbaa !6
  br i1 %12, label %16, label %17

16:                                               ; preds = %15, %16
  br label %16

17:                                               ; preds = %15, %13
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i32 0, ptr @e, align 4, !tbaa !6
  store i32 0, ptr @c, align 4, !tbaa !6
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp ne i32 %1, 0
  %3 = load i32, ptr @b, align 4
  %4 = icmp ne i32 %3, 0
  %5 = select i1 %2, i1 %4, i1 false
  %6 = zext i1 %5 to i32
  store i32 %6, ptr @k, align 4, !tbaa !6
  %7 = select i1 %5, i8 54, i8 0
  store i8 %7, ptr @j, align 4, !tbaa !10
  %8 = mul i8 %7, -109
  store i8 %8, ptr @g, align 4, !tbaa !10
  %9 = load i32, ptr @d, align 4, !tbaa !6
  %10 = icmp eq i32 %9, 0
  %11 = load i32, ptr @f, align 4, !tbaa !6
  %12 = icmp eq i32 %11, 0
  br i1 %10, label %13, label %15

13:                                               ; preds = %0
  store i32 9, ptr @i, align 4, !tbaa !6
  store i32 9, ptr @h, align 4, !tbaa !6
  br i1 %12, label %14, label %17

14:                                               ; preds = %13, %14
  br label %14

15:                                               ; preds = %0
  store i32 1, ptr @c, align 4, !tbaa !6
  br i1 %12, label %16, label %18

16:                                               ; preds = %15, %16
  br label %16

17:                                               ; preds = %13
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %15
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
