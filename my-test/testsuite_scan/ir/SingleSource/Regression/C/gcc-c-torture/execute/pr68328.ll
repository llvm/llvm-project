; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68328.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68328.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i32 1, align 4
@d = dso_local local_unnamed_addr global i32 1, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local noundef i32 @foo() local_unnamed_addr #0 {
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !6
  ret i32 4195552
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @bar(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  tail call void asm sideeffect "", "imr,imr,~{memory}"(i32 %0, i32 %1) #3, !srcloc !7
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void @abort() #4
  unreachable

5:                                                ; preds = %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local i32 @baz(i32 noundef %0) local_unnamed_addr #2 {
  %2 = tail call i32 @foo()
  %3 = load i32, ptr @c, align 4, !tbaa !8
  %4 = load i32, ptr @d, align 4
  %5 = load i32, ptr @a, align 4, !tbaa !8
  br label %6

6:                                                ; preds = %26, %1
  %7 = phi i32 [ %5, %1 ], [ %27, %26 ]
  %8 = phi i32 [ %4, %1 ], [ %28, %26 ]
  %9 = phi i32 [ %3, %1 ], [ %29, %26 ]
  %10 = phi i8 [ undef, %1 ], [ %13, %26 ]
  %11 = icmp eq i32 %9, 0
  %12 = trunc i32 %8 to i8
  %13 = select i1 %11, i8 %10, i8 %12
  %14 = zext i8 %13 to i32
  %15 = icmp sgt i32 %0, %14
  %16 = xor i8 %13, 1
  %17 = add i8 %16, -120
  %18 = icmp sgt i8 %17, 97
  %19 = and i1 %15, %18
  %20 = zext i1 %19 to i32
  %21 = icmp eq i32 %7, %20
  br i1 %21, label %26, label %22

22:                                               ; preds = %6
  tail call void @bar(i32 noundef 1193046, i32 noundef 0)
  %23 = load i32, ptr @c, align 4, !tbaa !8
  %24 = load i32, ptr @d, align 4
  %25 = load i32, ptr @a, align 4, !tbaa !8
  br label %26

26:                                               ; preds = %22, %6
  %27 = phi i32 [ %25, %22 ], [ %7, %6 ]
  %28 = phi i32 [ %24, %22 ], [ %8, %6 ]
  %29 = phi i32 [ %23, %22 ], [ %9, %6 ]
  %30 = load i32, ptr @b, align 4, !tbaa !8
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %32, label %6

32:                                               ; preds = %26
  %33 = load i32, ptr @e, align 4, !tbaa !8
  ret i32 %33
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call i32 @foo()
  %2 = load i32, ptr @c, align 4, !tbaa !8
  %3 = load i32, ptr @d, align 4
  %4 = load i32, ptr @a, align 4, !tbaa !8
  br label %5

5:                                                ; preds = %24, %0
  %6 = phi i32 [ %4, %0 ], [ %25, %24 ]
  %7 = phi i32 [ %3, %0 ], [ %26, %24 ]
  %8 = phi i32 [ %2, %0 ], [ %27, %24 ]
  %9 = phi i8 [ undef, %0 ], [ %12, %24 ]
  %10 = icmp eq i32 %8, 0
  %11 = trunc i32 %7 to i8
  %12 = select i1 %10, i8 %9, i8 %11
  %13 = icmp ult i8 %12, 2
  %14 = xor i8 %12, 1
  %15 = add i8 %14, -120
  %16 = icmp sgt i8 %15, 97
  %17 = and i1 %13, %16
  %18 = zext i1 %17 to i32
  %19 = icmp eq i32 %6, %18
  br i1 %19, label %24, label %20

20:                                               ; preds = %5
  tail call void @bar(i32 noundef 1193046, i32 noundef 0)
  %21 = load i32, ptr @c, align 4, !tbaa !8
  %22 = load i32, ptr @d, align 4
  %23 = load i32, ptr @a, align 4, !tbaa !8
  br label %24

24:                                               ; preds = %20, %5
  %25 = phi i32 [ %23, %20 ], [ %6, %5 ]
  %26 = phi i32 [ %22, %20 ], [ %7, %5 ]
  %27 = phi i32 [ %21, %20 ], [ %8, %5 ]
  %28 = load i32, ptr @b, align 4, !tbaa !8
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %5

30:                                               ; preds = %24
  ret i32 0
}

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 104}
!7 = !{i64 224}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
