; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56205.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56205.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@c = dso_local local_unnamed_addr global [128 x i8] zeroinitializer, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"baz\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"foo\00", align 1
@a = dso_local local_unnamed_addr global i32 0, align 4
@.str.3 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"bar\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local void @f4(i32 noundef %0, ptr noundef readonly captures(address_is_null) %1, ...) local_unnamed_addr #0 {
  %3 = alloca %struct.__va_list, align 8
  %4 = alloca %struct.__va_list, align 8
  %5 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  call void @llvm.va_start.p0(ptr nonnull %4)
  %6 = icmp eq i32 %0, 0
  %7 = load i8, ptr @c, align 4
  %8 = icmp eq i8 %7, 0
  %9 = select i1 %6, i1 %8, i1 false
  br i1 %9, label %10, label %13

10:                                               ; preds = %2
  %11 = load i32, ptr @b, align 4, !tbaa !6
  %12 = add nsw i32 %11, 1
  store i32 %12, ptr @b, align 4, !tbaa !6
  br label %13

13:                                               ; preds = %10, %2
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %4, i64 32, i1 false), !tbaa.struct !10
  %14 = select i1 %6, ptr @.str.1, ptr @.str.3
  %15 = load i32, ptr @a, align 4, !tbaa !6
  %16 = add nsw i32 %15, 1
  store i32 %16, ptr @a, align 4, !tbaa !6
  %17 = icmp eq ptr %1, null
  br i1 %17, label %25, label %18

18:                                               ; preds = %13
  %19 = load i8, ptr %1, align 1, !tbaa !13
  %20 = icmp eq i8 %19, 0
  %21 = select i1 %20, ptr @.str.3, ptr @.str.4
  call void (ptr, ...) @f1(ptr nonnull poison, ptr noundef nonnull %14, i32 noundef %16, ptr noundef nonnull %21)
  %22 = load i8, ptr %1, align 1, !tbaa !13
  %23 = icmp eq i8 %22, 0
  br i1 %23, label %26, label %24

24:                                               ; preds = %18
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull readonly align 8 dereferenceable(32) %5, i64 32, i1 false), !tbaa.struct !10
  call fastcc void @f2(ptr noundef readonly %1, ptr dead_on_return noundef %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  br label %26

25:                                               ; preds = %13
  call void (ptr, ...) @f1(ptr nonnull poison, ptr noundef nonnull %14, i32 noundef %16, ptr noundef nonnull @.str.3)
  br label %26

26:                                               ; preds = %18, %24, %25
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.va_end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #2

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  tail call void asm sideeffect "", "~{memory}"() #7, !srcloc !14
  tail call void (i32, ptr, ...) @f4(i32 noundef 0, ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, double noundef 1.200000e+01, i32 noundef 26)
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp ne i32 %1, 1
  %3 = load i32, ptr @b, align 4
  %4 = icmp ne i32 %3, 1
  %5 = select i1 %2, i1 true, i1 %4
  br i1 %5, label %6, label %7

6:                                                ; preds = %0
  tail call void @abort() #8
  unreachable

7:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #5

; Function Attrs: noinline nounwind uwtable
define internal void @f1(ptr readnone captures(none) %0, ...) unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  tail call void asm sideeffect "", "~{memory}"() #7, !srcloc !15
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %4 = load i32, ptr %3, align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %14, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 8
  store i32 %7, ptr %3, align 8
  %8 = icmp samesign ult i32 %4, -7
  br i1 %8, label %9, label %14

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %11 = load ptr, ptr %10, align 8
  %12 = sext i32 %4 to i64
  %13 = getelementptr inbounds i8, ptr %11, i64 %12
  br label %18

14:                                               ; preds = %6, %1
  %15 = phi i32 [ %7, %6 ], [ %4, %1 ]
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr %2, align 8
  br label %18

18:                                               ; preds = %14, %9
  %19 = phi i32 [ %7, %9 ], [ %15, %14 ]
  %20 = phi ptr [ %13, %9 ], [ %16, %14 ]
  %21 = load ptr, ptr %20, align 8, !tbaa !16
  %22 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %21, ptr noundef nonnull dereferenceable(4) @.str.1) #7
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %24, label %61

24:                                               ; preds = %18
  %25 = icmp sgt i32 %19, -1
  br i1 %25, label %34, label %26

26:                                               ; preds = %24
  %27 = add nsw i32 %19, 8
  store i32 %27, ptr %3, align 8
  %28 = icmp samesign ult i32 %19, -7
  br i1 %28, label %29, label %34

29:                                               ; preds = %26
  %30 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %31 = load ptr, ptr %30, align 8
  %32 = sext i32 %19 to i64
  %33 = getelementptr inbounds i8, ptr %31, i64 %32
  br label %38

34:                                               ; preds = %26, %24
  %35 = phi i32 [ %27, %26 ], [ %19, %24 ]
  %36 = load ptr, ptr %2, align 8
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 8
  store ptr %37, ptr %2, align 8
  br label %38

38:                                               ; preds = %34, %29
  %39 = phi i32 [ %27, %29 ], [ %35, %34 ]
  %40 = phi ptr [ %33, %29 ], [ %36, %34 ]
  %41 = load i32, ptr %40, align 8, !tbaa !6
  %42 = icmp eq i32 %41, 1
  br i1 %42, label %43, label %61

43:                                               ; preds = %38
  %44 = icmp sgt i32 %39, -1
  br i1 %44, label %53, label %45

45:                                               ; preds = %43
  %46 = add nsw i32 %39, 8
  store i32 %46, ptr %3, align 8
  %47 = icmp samesign ult i32 %39, -7
  br i1 %47, label %48, label %53

48:                                               ; preds = %45
  %49 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %50 = load ptr, ptr %49, align 8
  %51 = sext i32 %39 to i64
  %52 = getelementptr inbounds i8, ptr %50, i64 %51
  br label %56

53:                                               ; preds = %45, %43
  %54 = load ptr, ptr %2, align 8
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 8
  store ptr %55, ptr %2, align 8
  br label %56

56:                                               ; preds = %53, %48
  %57 = phi ptr [ %52, %48 ], [ %54, %53 ]
  %58 = load ptr, ptr %57, align 8, !tbaa !16
  %59 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %58, ptr noundef nonnull dereferenceable(4) @.str.4) #7
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %62, label %61

61:                                               ; preds = %56, %38, %18
  call void @abort() #8
  unreachable

62:                                               ; preds = %56
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: noinline nounwind uwtable
define internal fastcc void @f2(ptr noundef nonnull readonly captures(none) %0, ptr dead_on_return noundef nonnull captures(none) %1) unnamed_addr #0 {
  tail call void asm sideeffect "", "~{memory}"() #7, !srcloc !18
  %3 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %0, ptr noundef nonnull dereferenceable(4) @.str) #7
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %63

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %7 = load i32, ptr %6, align 8
  %8 = icmp sgt i32 %7, -1
  br i1 %8, label %17, label %9

9:                                                ; preds = %5
  %10 = add nsw i32 %7, 8
  store i32 %10, ptr %6, align 8
  %11 = icmp samesign ult i32 %7, -7
  br i1 %11, label %12, label %17

12:                                               ; preds = %9
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load ptr, ptr %13, align 8
  %15 = sext i32 %7 to i64
  %16 = getelementptr inbounds i8, ptr %14, i64 %15
  br label %21

17:                                               ; preds = %9, %5
  %18 = phi i32 [ %10, %9 ], [ %7, %5 ]
  %19 = load ptr, ptr %1, align 8
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 8
  store ptr %20, ptr %1, align 8
  br label %21

21:                                               ; preds = %17, %12
  %22 = phi i32 [ %10, %12 ], [ %18, %17 ]
  %23 = phi ptr [ %16, %12 ], [ %19, %17 ]
  %24 = load ptr, ptr %23, align 8, !tbaa !16
  %25 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %24, ptr noundef nonnull dereferenceable(4) @.str.1) #7
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %27, label %63

27:                                               ; preds = %21
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %29 = load i32, ptr %28, align 4
  %30 = icmp sgt i32 %29, -1
  br i1 %30, label %39, label %31

31:                                               ; preds = %27
  %32 = add nsw i32 %29, 16
  store i32 %32, ptr %28, align 4
  %33 = icmp samesign ult i32 %29, -15
  br i1 %33, label %34, label %39

34:                                               ; preds = %31
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %36 = load ptr, ptr %35, align 8
  %37 = sext i32 %29 to i64
  %38 = getelementptr inbounds i8, ptr %36, i64 %37
  br label %42

39:                                               ; preds = %31, %27
  %40 = load ptr, ptr %1, align 8
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 8
  store ptr %41, ptr %1, align 8
  br label %42

42:                                               ; preds = %39, %34
  %43 = phi ptr [ %38, %34 ], [ %40, %39 ]
  %44 = load double, ptr %43, align 8, !tbaa !19
  %45 = fcmp une double %44, 1.200000e+01
  br i1 %45, label %63, label %46

46:                                               ; preds = %42
  %47 = icmp sgt i32 %22, -1
  br i1 %47, label %56, label %48

48:                                               ; preds = %46
  %49 = add nsw i32 %22, 8
  store i32 %49, ptr %6, align 8
  %50 = icmp samesign ult i32 %22, -7
  br i1 %50, label %51, label %56

51:                                               ; preds = %48
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %53 = load ptr, ptr %52, align 8
  %54 = sext i32 %22 to i64
  %55 = getelementptr inbounds i8, ptr %53, i64 %54
  br label %59

56:                                               ; preds = %48, %46
  %57 = load ptr, ptr %1, align 8
  %58 = getelementptr inbounds nuw i8, ptr %57, i64 8
  store ptr %58, ptr %1, align 8
  br label %59

59:                                               ; preds = %56, %51
  %60 = phi ptr [ %55, %51 ], [ %57, %56 ]
  %61 = load i32, ptr %60, align 8, !tbaa !6
  %62 = icmp eq i32 %61, 26
  br i1 %62, label %64, label %63

63:                                               ; preds = %59, %42, %21, %2
  tail call void @abort() #8
  unreachable

64:                                               ; preds = %59
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #6

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

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
!10 = !{i64 0, i64 8, !11, i64 8, i64 8, !11, i64 16, i64 8, !11, i64 24, i64 4, !6, i64 28, i64 4, !6}
!11 = !{!12, !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!8, !8, i64 0}
!14 = !{i64 1173}
!15 = !{i64 186}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 omnipotent char", !12, i64 0}
!18 = !{i64 600}
!19 = !{!20, !20, i64 0}
!20 = !{!"double", !8, i64 0}
