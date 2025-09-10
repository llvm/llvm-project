; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071213-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071213-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

; Function Attrs: nofree nounwind uwtable
define dso_local void @h(i32 noundef %0, ptr dead_on_return noundef captures(none) %1) local_unnamed_addr #0 {
  switch i32 %0, label %81 [
    i32 1, label %3
    i32 5, label %42
  ]

3:                                                ; preds = %2
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %5 = load i32, ptr %4, align 8
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %3
  %8 = add nsw i32 %5, 8
  store i32 %8, ptr %4, align 8
  %9 = icmp samesign ult i32 %5, -7
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %19

15:                                               ; preds = %7, %3
  %16 = phi i32 [ %8, %7 ], [ %5, %3 ]
  %17 = load ptr, ptr %1, align 8
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %18, ptr %1, align 8
  br label %19

19:                                               ; preds = %15, %10
  %20 = phi i32 [ %8, %10 ], [ %16, %15 ]
  %21 = phi ptr [ %14, %10 ], [ %17, %15 ]
  %22 = load i32, ptr %21, align 8, !tbaa !6
  %23 = icmp eq i32 %22, 3
  br i1 %23, label %24, label %41

24:                                               ; preds = %19
  %25 = icmp sgt i32 %20, -1
  br i1 %25, label %34, label %26

26:                                               ; preds = %24
  %27 = add nsw i32 %20, 8
  store i32 %27, ptr %4, align 8
  %28 = icmp samesign ult i32 %20, -7
  br i1 %28, label %29, label %34

29:                                               ; preds = %26
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %31 = load ptr, ptr %30, align 8
  %32 = sext i32 %20 to i64
  %33 = getelementptr inbounds i8, ptr %31, i64 %32
  br label %37

34:                                               ; preds = %26, %24
  %35 = load ptr, ptr %1, align 8
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 8
  store ptr %36, ptr %1, align 8
  br label %37

37:                                               ; preds = %34, %29
  %38 = phi ptr [ %33, %29 ], [ %35, %34 ]
  %39 = load i32, ptr %38, align 8, !tbaa !6
  %40 = icmp eq i32 %39, 4
  br i1 %40, label %82, label %41

41:                                               ; preds = %37, %19
  tail call void @abort() #4
  unreachable

42:                                               ; preds = %2
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %44 = load i32, ptr %43, align 8
  %45 = icmp sgt i32 %44, -1
  br i1 %45, label %54, label %46

46:                                               ; preds = %42
  %47 = add nsw i32 %44, 8
  store i32 %47, ptr %43, align 8
  %48 = icmp samesign ult i32 %44, -7
  br i1 %48, label %49, label %54

49:                                               ; preds = %46
  %50 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %51 = load ptr, ptr %50, align 8
  %52 = sext i32 %44 to i64
  %53 = getelementptr inbounds i8, ptr %51, i64 %52
  br label %58

54:                                               ; preds = %46, %42
  %55 = phi i32 [ %47, %46 ], [ %44, %42 ]
  %56 = load ptr, ptr %1, align 8
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 8
  store ptr %57, ptr %1, align 8
  br label %58

58:                                               ; preds = %54, %49
  %59 = phi i32 [ %47, %49 ], [ %55, %54 ]
  %60 = phi ptr [ %53, %49 ], [ %56, %54 ]
  %61 = load i32, ptr %60, align 8, !tbaa !6
  %62 = icmp eq i32 %61, 9
  br i1 %62, label %63, label %80

63:                                               ; preds = %58
  %64 = icmp sgt i32 %59, -1
  br i1 %64, label %73, label %65

65:                                               ; preds = %63
  %66 = add nsw i32 %59, 8
  store i32 %66, ptr %43, align 8
  %67 = icmp samesign ult i32 %59, -7
  br i1 %67, label %68, label %73

68:                                               ; preds = %65
  %69 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %70 = load ptr, ptr %69, align 8
  %71 = sext i32 %59 to i64
  %72 = getelementptr inbounds i8, ptr %70, i64 %71
  br label %76

73:                                               ; preds = %65, %63
  %74 = load ptr, ptr %1, align 8
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 8
  store ptr %75, ptr %1, align 8
  br label %76

76:                                               ; preds = %73, %68
  %77 = phi ptr [ %72, %68 ], [ %74, %73 ]
  %78 = load i32, ptr %77, align 8, !tbaa !6
  %79 = icmp eq i32 %78, 10
  br i1 %79, label %82, label %80

80:                                               ; preds = %76, %58
  tail call void @abort() #4
  unreachable

81:                                               ; preds = %2
  tail call void @abort() #4
  unreachable

82:                                               ; preds = %76, %37
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @f1(i32 noundef %0, i64 noundef %1, ...) local_unnamed_addr #0 {
  %3 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = load ptr, ptr %3, align 8, !tbaa !10
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !10
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %8 = load i32, ptr %7, align 8, !tbaa !6
  switch i32 %0, label %59 [
    i32 1, label %9
    i32 5, label %34
  ]

9:                                                ; preds = %2
  %10 = icmp sgt i32 %8, -1
  br i1 %10, label %17, label %11

11:                                               ; preds = %9
  %12 = add nsw i32 %8, 8
  %13 = icmp samesign ult i32 %8, -7
  br i1 %13, label %14, label %17

14:                                               ; preds = %11
  %15 = sext i32 %8 to i64
  %16 = getelementptr inbounds i8, ptr %6, i64 %15
  br label %20

17:                                               ; preds = %11, %9
  %18 = phi i32 [ %12, %11 ], [ %8, %9 ]
  %19 = getelementptr inbounds nuw i8, ptr %4, i64 8
  br label %20

20:                                               ; preds = %17, %14
  %21 = phi ptr [ %19, %17 ], [ %4, %14 ]
  %22 = phi i32 [ %18, %17 ], [ %12, %14 ]
  %23 = phi ptr [ %4, %17 ], [ %16, %14 ]
  %24 = load i32, ptr %23, align 8, !tbaa !6
  %25 = icmp eq i32 %24, 3
  br i1 %25, label %26, label %33

26:                                               ; preds = %20
  %27 = icmp slt i32 %22, -7
  %28 = sext i32 %22 to i64
  %29 = getelementptr inbounds i8, ptr %6, i64 %28
  %30 = select i1 %27, ptr %29, ptr %21
  %31 = load i32, ptr %30, align 8, !tbaa !6
  %32 = icmp eq i32 %31, 4
  br i1 %32, label %60, label %33

33:                                               ; preds = %26, %20
  call void @abort() #4
  unreachable

34:                                               ; preds = %2
  %35 = icmp sgt i32 %8, -1
  br i1 %35, label %42, label %36

36:                                               ; preds = %34
  %37 = add nsw i32 %8, 8
  %38 = icmp samesign ult i32 %8, -7
  br i1 %38, label %39, label %42

39:                                               ; preds = %36
  %40 = sext i32 %8 to i64
  %41 = getelementptr inbounds i8, ptr %6, i64 %40
  br label %45

42:                                               ; preds = %36, %34
  %43 = phi i32 [ %37, %36 ], [ %8, %34 ]
  %44 = getelementptr inbounds nuw i8, ptr %4, i64 8
  br label %45

45:                                               ; preds = %42, %39
  %46 = phi ptr [ %44, %42 ], [ %4, %39 ]
  %47 = phi i32 [ %43, %42 ], [ %37, %39 ]
  %48 = phi ptr [ %4, %42 ], [ %41, %39 ]
  %49 = load i32, ptr %48, align 8, !tbaa !6
  %50 = icmp eq i32 %49, 9
  br i1 %50, label %51, label %58

51:                                               ; preds = %45
  %52 = icmp slt i32 %47, -7
  %53 = sext i32 %47 to i64
  %54 = getelementptr inbounds i8, ptr %6, i64 %53
  %55 = select i1 %52, ptr %54, ptr %46
  %56 = load i32, ptr %55, align 8, !tbaa !6
  %57 = icmp eq i32 %56, 10
  br i1 %57, label %62, label %58

58:                                               ; preds = %51, %45
  call void @abort() #4
  unreachable

59:                                               ; preds = %2
  call void @abort() #4
  unreachable

60:                                               ; preds = %26
  %61 = icmp eq i64 %1, 2
  br i1 %61, label %63, label %62

62:                                               ; preds = %51, %60
  call void @abort() #4
  unreachable

63:                                               ; preds = %60
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

; Function Attrs: nofree nounwind uwtable
define dso_local void @f2(i32 noundef %0, i32 noundef %1, i32 noundef %2, i64 noundef %3, ...) local_unnamed_addr #0 {
  %5 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  call void @llvm.va_start.p0(ptr nonnull %5)
  %6 = load ptr, ptr %5, align 8, !tbaa !10
  %7 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !10
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %10 = load i32, ptr %9, align 8, !tbaa !6
  switch i32 %0, label %61 [
    i32 1, label %11
    i32 5, label %36
  ]

11:                                               ; preds = %4
  %12 = icmp sgt i32 %10, -1
  br i1 %12, label %19, label %13

13:                                               ; preds = %11
  %14 = add nsw i32 %10, 8
  %15 = icmp samesign ult i32 %10, -7
  br i1 %15, label %16, label %19

16:                                               ; preds = %13
  %17 = sext i32 %10 to i64
  %18 = getelementptr inbounds i8, ptr %8, i64 %17
  br label %22

19:                                               ; preds = %13, %11
  %20 = phi i32 [ %14, %13 ], [ %10, %11 ]
  %21 = getelementptr inbounds nuw i8, ptr %6, i64 8
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi ptr [ %21, %19 ], [ %6, %16 ]
  %24 = phi i32 [ %20, %19 ], [ %14, %16 ]
  %25 = phi ptr [ %6, %19 ], [ %18, %16 ]
  %26 = load i32, ptr %25, align 8, !tbaa !6
  %27 = icmp eq i32 %26, 3
  br i1 %27, label %28, label %35

28:                                               ; preds = %22
  %29 = icmp slt i32 %24, -7
  %30 = sext i32 %24 to i64
  %31 = getelementptr inbounds i8, ptr %8, i64 %30
  %32 = select i1 %29, ptr %31, ptr %23
  %33 = load i32, ptr %32, align 8, !tbaa !6
  %34 = icmp eq i32 %33, 4
  br i1 %34, label %62, label %35

35:                                               ; preds = %28, %22
  call void @abort() #4
  unreachable

36:                                               ; preds = %4
  %37 = icmp sgt i32 %10, -1
  br i1 %37, label %44, label %38

38:                                               ; preds = %36
  %39 = add nsw i32 %10, 8
  %40 = icmp samesign ult i32 %10, -7
  br i1 %40, label %41, label %44

41:                                               ; preds = %38
  %42 = sext i32 %10 to i64
  %43 = getelementptr inbounds i8, ptr %8, i64 %42
  br label %47

44:                                               ; preds = %38, %36
  %45 = phi i32 [ %39, %38 ], [ %10, %36 ]
  %46 = getelementptr inbounds nuw i8, ptr %6, i64 8
  br label %47

47:                                               ; preds = %44, %41
  %48 = phi ptr [ %46, %44 ], [ %6, %41 ]
  %49 = phi i32 [ %45, %44 ], [ %39, %41 ]
  %50 = phi ptr [ %6, %44 ], [ %43, %41 ]
  %51 = load i32, ptr %50, align 8, !tbaa !6
  %52 = icmp eq i32 %51, 9
  br i1 %52, label %53, label %60

53:                                               ; preds = %47
  %54 = icmp slt i32 %49, -7
  %55 = sext i32 %49 to i64
  %56 = getelementptr inbounds i8, ptr %8, i64 %55
  %57 = select i1 %54, ptr %56, ptr %48
  %58 = load i32, ptr %57, align 8, !tbaa !6
  %59 = icmp eq i32 %58, 10
  br i1 %59, label %62, label %60

60:                                               ; preds = %53, %47
  call void @abort() #4
  unreachable

61:                                               ; preds = %4
  call void @abort() #4
  unreachable

62:                                               ; preds = %28, %53
  %63 = icmp ne i32 %0, 5
  %64 = icmp ne i32 %1, 6
  %65 = or i1 %63, %64
  %66 = icmp ne i32 %2, 7
  %67 = or i1 %65, %66
  %68 = icmp ne i64 %3, 8
  %69 = or i1 %67, %68
  br i1 %69, label %70, label %71

70:                                               ; preds = %62
  call void @abort() #4
  unreachable

71:                                               ; preds = %62
  call void @llvm.va_end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void (i32, i64, ...) @f1(i32 noundef 1, i64 noundef 2, i32 noundef 3, i32 noundef 4)
  tail call void (i32, i32, i32, i64, ...) @f2(i32 noundef 5, i32 noundef 6, i32 noundef 7, i64 noundef 8, i32 noundef 9, i32 noundef 10)
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"any pointer", !8, i64 0}
