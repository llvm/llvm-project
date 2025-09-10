; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050502-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050502-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [10 x i8] c"abcde'fgh\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"fgh\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"abcde\00", align 1
@.str.3 = private unnamed_addr constant [11 x i8] c"ABCDEFG\22HI\00", align 1
@.str.5 = private unnamed_addr constant [8 x i8] c"ABCDEFG\00", align 1
@.str.6 = private unnamed_addr constant [11 x i8] c"abcd\22e'fgh\00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c"e'fgh\00", align 1
@.str.8 = private unnamed_addr constant [5 x i8] c"abcd\00", align 1
@.str.9 = private unnamed_addr constant [12 x i8] c"ABCDEF'G\22HI\00", align 1
@.str.10 = private unnamed_addr constant [5 x i8] c"G\22HI\00", align 1
@.str.11 = private unnamed_addr constant [7 x i8] c"ABCDEF\00", align 1
@.str.12 = private unnamed_addr constant [10 x i8] c"abcdef@gh\00", align 1
@.str.14 = private unnamed_addr constant [7 x i8] c"abcdef\00", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 256) i32 @bar(ptr noundef captures(none) %0) local_unnamed_addr #0 {
  %2 = load ptr, ptr %0, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 1
  store ptr %3, ptr %0, align 8, !tbaa !6
  %4 = load i8, ptr %2, align 1, !tbaa !11
  %5 = zext i8 %4 to i32
  ret i32 %5
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @baz(i32 noundef %0) local_unnamed_addr #1 {
  %2 = icmp ne i32 %0, 64
  %3 = zext i1 %2 to i32
  ret i32 %3
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @foo(ptr noundef captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 1)) %1, i1 noundef %2, i1 noundef %3) local_unnamed_addr #2 {
  %5 = tail call i32 @bar(ptr noundef %0)
  %6 = trunc nuw i32 %5 to i8
  store i8 %6, ptr %1, align 1, !tbaa !11
  %7 = tail call i32 @bar(ptr noundef %0)
  %8 = icmp eq i32 %7, 39
  %9 = select i1 %2, i1 %8, i1 false
  %10 = icmp eq i32 %7, 34
  %11 = select i1 %3, i1 %10, i1 false
  %12 = select i1 %9, i1 true, i1 %11
  br i1 %12, label %39, label %13

13:                                               ; preds = %4
  %14 = or i1 %2, %3
  br i1 %14, label %18, label %15

15:                                               ; preds = %13
  %16 = tail call i32 @baz(i32 noundef %7)
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %39, label %30

18:                                               ; preds = %13, %18
  %19 = phi i64 [ %22, %18 ], [ 1, %13 ]
  %20 = phi i32 [ %24, %18 ], [ %7, %13 ]
  %21 = trunc nuw i32 %20 to i8
  %22 = add nuw nsw i64 %19, 1
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 %19
  store i8 %21, ptr %23, align 1, !tbaa !11
  %24 = tail call i32 @bar(ptr noundef %0)
  %25 = icmp eq i32 %24, 39
  %26 = select i1 %2, i1 %25, i1 false
  %27 = icmp eq i32 %24, 34
  %28 = select i1 %3, i1 %27, i1 false
  %29 = select i1 %26, i1 true, i1 %28
  br i1 %29, label %39, label %18

30:                                               ; preds = %15, %30
  %31 = phi i32 [ %36, %30 ], [ %7, %15 ]
  %32 = phi i64 [ %34, %30 ], [ 1, %15 ]
  %33 = trunc nuw i32 %31 to i8
  %34 = add nuw nsw i64 %32, 1
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 %32
  store i8 %33, ptr %35, align 1, !tbaa !11
  %36 = tail call i32 @bar(ptr noundef %0)
  %37 = tail call i32 @baz(i32 noundef %36)
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %39, label %30

39:                                               ; preds = %30, %18, %15, %4
  %40 = phi i64 [ 1, %4 ], [ 1, %15 ], [ %22, %18 ], [ %34, %30 ]
  %41 = and i64 %40, 4294967295
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 %41
  store i8 0, ptr %42, align 1, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = alloca [64 x i8], align 1
  %2 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  store ptr @.str, ptr %2, align 8, !tbaa !6
  call void @foo(ptr noundef nonnull %2, ptr noundef nonnull %1, i1 noundef true, i1 noundef false)
  %3 = load ptr, ptr %2, align 8, !tbaa !6
  %4 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %3, ptr noundef nonnull dereferenceable(4) @.str.1) #8
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %9

6:                                                ; preds = %0
  %7 = call i32 @bcmp(ptr noundef nonnull dereferenceable(6) %1, ptr noundef nonnull dereferenceable(6) @.str.2, i64 6)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %6, %0
  tail call void @abort() #9
  unreachable

10:                                               ; preds = %6
  store ptr @.str.3, ptr %2, align 8, !tbaa !6
  call void @foo(ptr noundef nonnull %2, ptr noundef nonnull %1, i1 noundef false, i1 noundef true)
  %11 = load ptr, ptr %2, align 8, !tbaa !6
  %12 = load i8, ptr %11, align 1
  %13 = icmp eq i8 %12, 72
  br i1 %13, label %14, label %25

14:                                               ; preds = %10
  %15 = getelementptr inbounds nuw i8, ptr %11, i64 1
  %16 = load i8, ptr %15, align 1
  %17 = icmp eq i8 %16, 73
  br i1 %17, label %18, label %25

18:                                               ; preds = %14
  %19 = getelementptr inbounds nuw i8, ptr %11, i64 2
  %20 = load i8, ptr %19, align 1
  %21 = icmp eq i8 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %18
  %23 = call i32 @bcmp(ptr noundef nonnull dereferenceable(8) %1, ptr noundef nonnull dereferenceable(8) @.str.5, i64 8)
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %26, label %25

25:                                               ; preds = %14, %10, %22, %18
  tail call void @abort() #9
  unreachable

26:                                               ; preds = %22
  store ptr @.str.6, ptr %2, align 8, !tbaa !6
  call void @foo(ptr noundef nonnull %2, ptr noundef nonnull %1, i1 noundef true, i1 noundef true)
  %27 = load ptr, ptr %2, align 8, !tbaa !6
  %28 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %27, ptr noundef nonnull dereferenceable(6) @.str.7) #8
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %33

30:                                               ; preds = %26
  %31 = call i32 @bcmp(ptr noundef nonnull dereferenceable(5) %1, ptr noundef nonnull dereferenceable(5) @.str.8, i64 5)
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %34, label %33

33:                                               ; preds = %30, %26
  tail call void @abort() #9
  unreachable

34:                                               ; preds = %30
  store ptr @.str.9, ptr %2, align 8, !tbaa !6
  call void @foo(ptr noundef nonnull %2, ptr noundef nonnull %1, i1 noundef true, i1 noundef true)
  %35 = load ptr, ptr %2, align 8, !tbaa !6
  %36 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %35, ptr noundef nonnull dereferenceable(5) @.str.10) #8
  %37 = icmp eq i32 %36, 0
  br i1 %37, label %38, label %41

38:                                               ; preds = %34
  %39 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %1, ptr noundef nonnull dereferenceable(7) @.str.11, i64 7)
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %42, label %41

41:                                               ; preds = %38, %34
  tail call void @abort() #9
  unreachable

42:                                               ; preds = %38
  store ptr @.str.12, ptr %2, align 8, !tbaa !6
  call void @foo(ptr noundef nonnull %2, ptr noundef nonnull %1, i1 noundef false, i1 noundef false)
  %43 = load ptr, ptr %2, align 8, !tbaa !6
  %44 = load i8, ptr %43, align 1
  %45 = icmp eq i8 %44, 103
  br i1 %45, label %46, label %57

46:                                               ; preds = %42
  %47 = getelementptr inbounds nuw i8, ptr %43, i64 1
  %48 = load i8, ptr %47, align 1
  %49 = icmp eq i8 %48, 104
  br i1 %49, label %50, label %57

50:                                               ; preds = %46
  %51 = getelementptr inbounds nuw i8, ptr %43, i64 2
  %52 = load i8, ptr %51, align 1
  %53 = icmp eq i8 %52, 0
  br i1 %53, label %54, label %57

54:                                               ; preds = %50
  %55 = call i32 @bcmp(ptr noundef nonnull dereferenceable(7) %1, ptr noundef nonnull dereferenceable(7) @.str.14, i64 7)
  %56 = icmp eq i32 %55, 0
  br i1 %56, label %58, label %57

57:                                               ; preds = %46, %42, %54, %50
  tail call void @abort() #9
  unreachable

58:                                               ; preds = %54
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #8
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #5

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #6

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #7

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #8 = { nounwind }
attributes #9 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!9, !9, i64 0}
