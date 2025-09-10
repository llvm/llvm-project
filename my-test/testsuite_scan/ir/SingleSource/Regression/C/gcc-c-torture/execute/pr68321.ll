; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68321.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68321.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@e = dso_local local_unnamed_addr global i32 1, align 4
@u = dso_local local_unnamed_addr global i32 5, align 4
@t5 = dso_local local_unnamed_addr global i32 0, align 4
@n = dso_local local_unnamed_addr global i8 0, align 1
@t2 = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@m = dso_local local_unnamed_addr global i32 0, align 4
@t = dso_local local_unnamed_addr global i8 0, align 4
@a = dso_local local_unnamed_addr global [1 x i32] zeroinitializer, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@k = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @fn1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add i32 %0, -3
  %3 = icmp ult i32 %2, -5
  %4 = load i32, ptr @t5, align 4
  %5 = icmp ne i32 %4, 0
  %6 = select i1 %3, i1 %5, i1 false
  br i1 %6, label %12, label %7

7:                                                ; preds = %1
  %8 = load i32, ptr @b, align 4
  %9 = icmp eq i32 %8, -1
  %10 = and i32 %0, 4
  store i32 %10, ptr @t2, align 4, !tbaa !6
  br i1 %9, label %11, label %13

11:                                               ; preds = %7, %11
  br label %11

12:                                               ; preds = %1, %12
  br label %12

13:                                               ; preds = %7
  ret i32 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @e, align 4, !tbaa !6
  %2 = icmp sgt i32 %1, -1
  br i1 %2, label %5, label %3

3:                                                ; preds = %0
  %4 = load i32, ptr @t2, align 4, !tbaa !6
  br label %33

5:                                                ; preds = %0
  %6 = load i32, ptr @m, align 4, !tbaa !6
  %7 = freeze i32 %6
  %8 = icmp eq i32 %7, 0
  %9 = load i8, ptr @t, align 4
  %10 = load i32, ptr @t5, align 4
  %11 = icmp ne i32 %10, 0
  %12 = load i32, ptr @b, align 4
  %13 = freeze i32 %12
  %14 = icmp eq i32 %13, -1
  br i1 %14, label %15, label %23

15:                                               ; preds = %5
  %16 = select i1 %8, i8 %9, i8 undef
  %17 = icmp ugt i8 %16, 2
  %18 = select i1 %17, i1 %11, i1 false
  br i1 %18, label %19, label %20

19:                                               ; preds = %24, %15
  br label %32

20:                                               ; preds = %15
  %21 = and i8 %16, 4
  %22 = zext nneg i8 %21 to i32
  store i32 %22, ptr @t2, align 4, !tbaa !6
  br label %31

23:                                               ; preds = %5
  br i1 %8, label %24, label %30

24:                                               ; preds = %23
  %25 = icmp ugt i8 %9, 2
  %26 = select i1 %25, i1 %11, i1 false
  br i1 %26, label %19, label %27

27:                                               ; preds = %24
  %28 = and i8 %9, 4
  %29 = zext nneg i8 %28 to i32
  store i32 %29, ptr @t2, align 4, !tbaa !6
  store i32 -1, ptr @e, align 4, !tbaa !6
  br label %33

30:                                               ; preds = %23
  store i32 0, ptr @t2, align 4, !tbaa !6
  store i32 -1, ptr @e, align 4, !tbaa !6
  br label %33

31:                                               ; preds = %20, %31
  br label %31

32:                                               ; preds = %19, %32
  br label %32

33:                                               ; preds = %3, %30, %27
  %34 = phi i32 [ %4, %3 ], [ 0, %30 ], [ %29, %27 ]
  %35 = sext i32 %34 to i64
  %36 = getelementptr inbounds i32, ptr @a, i64 %35
  %37 = load i32, ptr %36, align 4, !tbaa !6
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %33
  tail call void @abort() #3
  unreachable

40:                                               ; preds = %33
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
