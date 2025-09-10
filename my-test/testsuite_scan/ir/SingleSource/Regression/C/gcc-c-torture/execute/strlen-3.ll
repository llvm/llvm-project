; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/strlen-3.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/strlen-3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v0 = dso_local local_unnamed_addr global i32 0, align 4
@v1 = dso_local global i32 1, align 4
@v2 = dso_local global i32 2, align 4
@v3 = dso_local global i32 3, align 4
@v4 = dso_local global i32 4, align 4
@v5 = dso_local global i32 5, align 4
@v6 = dso_local global i32 6, align 4
@v7 = dso_local global i32 7, align 4
@a = internal constant [2 x [3 x [9 x i8]]] [[3 x [9 x i8]] [[9 x i8] c"1\00\00\00\00\00\00\00\00", [9 x i8] c"1\002\00\00\00\00\00\00", [9 x i8] zeroinitializer], [3 x [9 x i8]] [[9 x i8] c"12\003\00\00\00\00\00", [9 x i8] c"123\004\00\00\00\00", [9 x i8] zeroinitializer]], align 1
@.str = private unnamed_addr constant [26 x i8] c"assertion on line %i: %s\0A\00", align 1
@.str.30 = private unnamed_addr constant [34 x i8] c"strlen (&a[i0][i0][i0] + v1) == 0\00", align 1
@.str.31 = private unnamed_addr constant [34 x i8] c"strlen (&a[i0][i0][i0] + v2) == 0\00", align 1
@.str.32 = private unnamed_addr constant [34 x i8] c"strlen (&a[i0][i0][i0] + v7) == 0\00", align 1
@.str.33 = private unnamed_addr constant [34 x i8] c"strlen (&a[i0][i1][i0] + v1) == 0\00", align 1
@.str.34 = private unnamed_addr constant [34 x i8] c"strlen (&a[i0][i1][i0] + v2) == 1\00", align 1
@.str.35 = private unnamed_addr constant [34 x i8] c"strlen (&a[i0][i1][i0] + v3) == 0\00", align 1
@.str.36 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i0][i0] + v1) == 1\00", align 1
@.str.37 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i1][i0] + v1) == 2\00", align 1
@.str.38 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i1][i0] + v2) == 1\00", align 1
@.str.39 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i1][i0] + v3) == 0\00", align 1
@.str.40 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i1][i0] + v4) == 1\00", align 1
@.str.41 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i1][i0] + v5) == 0\00", align 1
@.str.42 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i1][i0] + v6) == 0\00", align 1
@.str.43 = private unnamed_addr constant [34 x i8] c"strlen (&a[i1][i1][i0] + v7) == 0\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_array_ref() local_unnamed_addr #0 {
  %1 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %2 = sext i32 %1 to i64
  %3 = getelementptr inbounds i8, ptr @a, i64 %2
  %4 = load i8, ptr %3, align 1
  %5 = icmp eq i8 %4, 0
  br i1 %5, label %8, label %6

6:                                                ; preds = %0
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 111, ptr noundef nonnull @.str.30) #4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %0
  %9 = load volatile i32, ptr @v2, align 4, !tbaa !6
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds i8, ptr @a, i64 %10
  %12 = load i8, ptr %11, align 1
  %13 = icmp eq i8 %12, 0
  br i1 %13, label %16, label %14

14:                                               ; preds = %8
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 112, ptr noundef nonnull @.str.31) #4
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %8
  %17 = load volatile i32, ptr @v7, align 4, !tbaa !6
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds i8, ptr @a, i64 %18
  %20 = load i8, ptr %19, align 1
  %21 = icmp eq i8 %20, 0
  br i1 %21, label %24, label %22

22:                                               ; preds = %16
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 113, ptr noundef nonnull @.str.32) #4
  tail call void @abort() #5
  unreachable

24:                                               ; preds = %16
  %25 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 9), i64 %26
  %28 = load i8, ptr %27, align 1
  %29 = icmp eq i8 %28, 0
  br i1 %29, label %32, label %30

30:                                               ; preds = %24
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 115, ptr noundef nonnull @.str.33) #4
  tail call void @abort() #5
  unreachable

32:                                               ; preds = %24
  %33 = load volatile i32, ptr @v2, align 4, !tbaa !6
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 9), i64 %34
  %36 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %35) #4
  %37 = icmp eq i64 %36, 1
  br i1 %37, label %40, label %38

38:                                               ; preds = %32
  %39 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 116, ptr noundef nonnull @.str.34) #4
  tail call void @abort() #5
  unreachable

40:                                               ; preds = %32
  %41 = load volatile i32, ptr @v3, align 4, !tbaa !6
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 9), i64 %42
  %44 = load i8, ptr %43, align 1
  %45 = icmp eq i8 %44, 0
  br i1 %45, label %48, label %46

46:                                               ; preds = %40
  %47 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 117, ptr noundef nonnull @.str.35) #4
  tail call void @abort() #5
  unreachable

48:                                               ; preds = %40
  %49 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 27), i64 %50
  %52 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %51) #4
  %53 = icmp eq i64 %52, 1
  br i1 %53, label %56, label %54

54:                                               ; preds = %48
  %55 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 119, ptr noundef nonnull @.str.36) #4
  tail call void @abort() #5
  unreachable

56:                                               ; preds = %48
  %57 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %58 = sext i32 %57 to i64
  %59 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 36), i64 %58
  %60 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %59) #4
  %61 = icmp eq i64 %60, 2
  br i1 %61, label %64, label %62

62:                                               ; preds = %56
  %63 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 120, ptr noundef nonnull @.str.37) #4
  tail call void @abort() #5
  unreachable

64:                                               ; preds = %56
  %65 = load volatile i32, ptr @v2, align 4, !tbaa !6
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 36), i64 %66
  %68 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %67) #4
  %69 = icmp eq i64 %68, 1
  br i1 %69, label %72, label %70

70:                                               ; preds = %64
  %71 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 121, ptr noundef nonnull @.str.38) #4
  tail call void @abort() #5
  unreachable

72:                                               ; preds = %64
  %73 = load volatile i32, ptr @v3, align 4, !tbaa !6
  %74 = sext i32 %73 to i64
  %75 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 36), i64 %74
  %76 = load i8, ptr %75, align 1
  %77 = icmp eq i8 %76, 0
  br i1 %77, label %80, label %78

78:                                               ; preds = %72
  %79 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 122, ptr noundef nonnull @.str.39) #4
  tail call void @abort() #5
  unreachable

80:                                               ; preds = %72
  %81 = load volatile i32, ptr @v4, align 4, !tbaa !6
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 36), i64 %82
  %84 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %83) #4
  %85 = icmp eq i64 %84, 1
  br i1 %85, label %88, label %86

86:                                               ; preds = %80
  %87 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 123, ptr noundef nonnull @.str.40) #4
  tail call void @abort() #5
  unreachable

88:                                               ; preds = %80
  %89 = load volatile i32, ptr @v5, align 4, !tbaa !6
  %90 = sext i32 %89 to i64
  %91 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 36), i64 %90
  %92 = load i8, ptr %91, align 1
  %93 = icmp eq i8 %92, 0
  br i1 %93, label %96, label %94

94:                                               ; preds = %88
  %95 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 124, ptr noundef nonnull @.str.41) #4
  tail call void @abort() #5
  unreachable

96:                                               ; preds = %88
  %97 = load volatile i32, ptr @v6, align 4, !tbaa !6
  %98 = sext i32 %97 to i64
  %99 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 36), i64 %98
  %100 = load i8, ptr %99, align 1
  %101 = icmp eq i8 %100, 0
  br i1 %101, label %104, label %102

102:                                              ; preds = %96
  %103 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 125, ptr noundef nonnull @.str.42) #4
  tail call void @abort() #5
  unreachable

104:                                              ; preds = %96
  %105 = load volatile i32, ptr @v7, align 4, !tbaa !6
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 36), i64 %106
  %108 = load i8, ptr %107, align 1
  %109 = icmp eq i8 %108, 0
  br i1 %109, label %112, label %110

110:                                              ; preds = %104
  %111 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 126, ptr noundef nonnull @.str.43) #4
  tail call void @abort() #5
  unreachable

112:                                              ; preds = %104
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void @test_array_ref()
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

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
