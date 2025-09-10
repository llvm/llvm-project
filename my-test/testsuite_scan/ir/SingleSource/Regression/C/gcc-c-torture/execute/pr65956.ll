; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65956.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65956.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v = dso_local global [3 x i8] zeroinitializer, align 1
@.str = private unnamed_addr constant [2 x i8] c"+\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"-\00", align 1
@__const.main.a = private unnamed_addr constant [3 x { ptr, i32, [4 x i8], i64 }] [{ ptr, i32, [4 x i8], i64 } { ptr getelementptr (i8, ptr @v, i64 1), i32 1, [4 x i8] zeroinitializer, i64 1 }, { ptr, i32, [4 x i8], i64 } { ptr @v, i32 0, [4 x i8] zeroinitializer, i64 0 }, { ptr, i32, [4 x i8], i64 } { ptr getelementptr (i8, ptr @v, i64 2), i32 2, [4 x i8] zeroinitializer, i64 2 }], align 8

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @fn1(ptr noundef readnone captures(address) %0, ptr noundef readnone captures(address) %1) local_unnamed_addr #0 {
  %3 = icmp ne ptr %0, getelementptr inbounds nuw (i8, ptr @v, i64 1)
  %4 = icmp ne ptr %1, getelementptr inbounds nuw (i8, ptr @v, i64 2)
  %5 = select i1 %3, i1 true, i1 %4
  br i1 %5, label %6, label %7

6:                                                ; preds = %2
  tail call void @abort() #4
  unreachable

7:                                                ; preds = %2
  %8 = load i8, ptr getelementptr inbounds nuw (i8, ptr @v, i64 1), align 1, !tbaa !6
  %9 = add i8 %8, 1
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @v, i64 1), align 1, !tbaa !6
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: noinline nounwind uwtable
define dso_local range(i32 0, 2) i32 @fn2(ptr noundef %0) local_unnamed_addr #2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !9
  call void asm sideeffect "", "=*imr,0,~{memory}"(ptr nonnull elementtype(ptr) %2, ptr %0) #5, !srcloc !12
  %3 = load ptr, ptr %2, align 8, !tbaa !9
  %4 = icmp eq ptr %3, @v
  %5 = zext i1 %4 to i32
  ret i32 %5
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @fn3(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i8, ptr %0, align 1, !tbaa !6
  %3 = icmp eq i8 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @abort() #4
  unreachable

5:                                                ; preds = %1
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #2 {
  switch i32 %0, label %41 [
    i32 219, label %3
    i32 220, label %22
  ]

3:                                                ; preds = %2
  %4 = getelementptr inbounds i8, ptr %1, i64 -48
  %5 = load ptr, ptr %4, align 8, !tbaa !9
  %6 = getelementptr inbounds i8, ptr %1, i64 -40
  %7 = load i32, ptr %6, align 8, !tbaa !13
  %8 = load ptr, ptr %1, align 8, !tbaa !9
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load i32, ptr %9, align 8, !tbaa !13
  %11 = icmp eq i32 %7, 0
  %12 = icmp eq i32 %10, 0
  %13 = select i1 %11, i1 true, i1 %12
  br i1 %13, label %22, label %14

14:                                               ; preds = %3
  %15 = tail call i32 @fn2(ptr noundef %5), !noalias !15
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %21, label %17

17:                                               ; preds = %14
  %18 = tail call i32 @fn2(ptr noundef %8), !noalias !15
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %21, label %20

20:                                               ; preds = %17
  tail call void @fn3(ptr noundef nonnull @.str), !noalias !15
  br label %21

21:                                               ; preds = %20, %17, %14
  tail call void @fn1(ptr noundef %5, ptr noundef %8), !noalias !15
  br label %22

22:                                               ; preds = %21, %3, %2
  %23 = getelementptr inbounds i8, ptr %1, i64 -48
  %24 = load ptr, ptr %23, align 8, !tbaa !9
  %25 = getelementptr inbounds i8, ptr %1, i64 -40
  %26 = load i32, ptr %25, align 8, !tbaa !13
  %27 = load ptr, ptr %1, align 8, !tbaa !9
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %29 = load i32, ptr %28, align 8, !tbaa !13
  %30 = icmp eq i32 %26, 0
  %31 = icmp eq i32 %29, 0
  %32 = select i1 %30, i1 true, i1 %31
  br i1 %32, label %41, label %33

33:                                               ; preds = %22
  %34 = tail call i32 @fn2(ptr noundef %24), !noalias !18
  %35 = icmp eq i32 %34, 0
  br i1 %35, label %40, label %36

36:                                               ; preds = %33
  %37 = tail call i32 @fn2(ptr noundef %27), !noalias !18
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %36
  tail call void @fn3(ptr noundef nonnull @.str.1), !noalias !18
  br label %40

40:                                               ; preds = %39, %36, %33
  tail call void @fn1(ptr noundef %24, ptr noundef %27), !noalias !18
  br label %41

41:                                               ; preds = %40, %22, %2
  ret i32 undef
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = tail call i32 @bar(i32 noundef 220, ptr noundef nonnull getelementptr inbounds nuw (i8, ptr @__const.main.a, i64 48))
  %2 = load i8, ptr getelementptr inbounds nuw (i8, ptr @v, i64 1), align 1, !tbaa !6
  %3 = icmp eq i8 %2, 1
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

5:                                                ; preds = %0
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"p1 omnipotent char", !11, i64 0}
!11 = !{!"any pointer", !7, i64 0}
!12 = !{i64 285}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !7, i64 0}
!15 = !{!16}
!16 = distinct !{!16, !17, !"foo: argument 0"}
!17 = distinct !{!17, !"foo"}
!18 = !{!19}
!19 = distinct !{!19, !20, !"foo: argument 0"}
!20 = distinct !{!20, !"foo"}
