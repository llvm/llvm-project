; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44942.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44942.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

; Function Attrs: nofree nounwind uwtable
define dso_local void @test1(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, fp128 %7, ...) local_unnamed_addr #0 {
  %9 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #4
  call void @llvm.va_start.p0(ptr nonnull %9)
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 24
  %11 = load i32, ptr %10, align 8
  %12 = icmp sgt i32 %11, -1
  br i1 %12, label %21, label %13

13:                                               ; preds = %8
  %14 = add nsw i32 %11, 8
  store i32 %14, ptr %10, align 8
  %15 = icmp samesign ult i32 %11, -7
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %18 = load ptr, ptr %17, align 8
  %19 = sext i32 %11 to i64
  %20 = getelementptr inbounds i8, ptr %18, i64 %19
  br label %24

21:                                               ; preds = %13, %8
  %22 = load ptr, ptr %9, align 8
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 8
  store ptr %23, ptr %9, align 8
  br label %24

24:                                               ; preds = %21, %16
  %25 = phi ptr [ %20, %16 ], [ %22, %21 ]
  %26 = load i32, ptr %25, align 8, !tbaa !6
  %27 = icmp eq i32 %26, 1234
  br i1 %27, label %29, label %28

28:                                               ; preds = %24
  call void @abort() #5
  unreachable

29:                                               ; preds = %24
  call void @llvm.va_end.p0(ptr nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test2(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, fp128 %7, i32 %8, fp128 %9, i32 %10, fp128 %11, i32 %12, fp128 %13, ...) local_unnamed_addr #0 {
  %15 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %15) #4
  call void @llvm.va_start.p0(ptr nonnull %15)
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 24
  %17 = load i32, ptr %16, align 8
  %18 = icmp sgt i32 %17, -1
  br i1 %18, label %27, label %19

19:                                               ; preds = %14
  %20 = add nsw i32 %17, 8
  store i32 %20, ptr %16, align 8
  %21 = icmp samesign ult i32 %17, -7
  br i1 %21, label %22, label %27

22:                                               ; preds = %19
  %23 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %24 = load ptr, ptr %23, align 8
  %25 = sext i32 %17 to i64
  %26 = getelementptr inbounds i8, ptr %24, i64 %25
  br label %30

27:                                               ; preds = %19, %14
  %28 = load ptr, ptr %15, align 8
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 8
  store ptr %29, ptr %15, align 8
  br label %30

30:                                               ; preds = %27, %22
  %31 = phi ptr [ %26, %22 ], [ %28, %27 ]
  %32 = load i32, ptr %31, align 8, !tbaa !6
  %33 = icmp eq i32 %32, 1234
  br i1 %33, label %35, label %34

34:                                               ; preds = %30
  call void @abort() #5
  unreachable

35:                                               ; preds = %30
  call void @llvm.va_end.p0(ptr nonnull %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #4
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test3(double %0, double %1, double %2, double %3, double %4, double %5, double %6, fp128 %7, ...) local_unnamed_addr #0 {
  %9 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #4
  call void @llvm.va_start.p0(ptr nonnull %9)
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 28
  %11 = load i32, ptr %10, align 4
  %12 = icmp sgt i32 %11, -1
  br i1 %12, label %21, label %13

13:                                               ; preds = %8
  %14 = add nsw i32 %11, 16
  store i32 %14, ptr %10, align 4
  %15 = icmp samesign ult i32 %11, -15
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %18 = load ptr, ptr %17, align 8
  %19 = sext i32 %11 to i64
  %20 = getelementptr inbounds i8, ptr %18, i64 %19
  br label %24

21:                                               ; preds = %13, %8
  %22 = load ptr, ptr %9, align 8
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 8
  store ptr %23, ptr %9, align 8
  br label %24

24:                                               ; preds = %21, %16
  %25 = phi ptr [ %20, %16 ], [ %22, %21 ]
  %26 = load double, ptr %25, align 8, !tbaa !10
  %27 = fcmp une double %26, 1.234000e+03
  br i1 %27, label %28, label %29

28:                                               ; preds = %24
  call void @abort() #5
  unreachable

29:                                               ; preds = %24
  call void @llvm.va_end.p0(ptr nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #4
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test4(double %0, double %1, double %2, double %3, double %4, double %5, double %6, fp128 %7, double %8, fp128 %9, double %10, fp128 %11, double %12, fp128 %13, ...) local_unnamed_addr #0 {
  %15 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %15) #4
  call void @llvm.va_start.p0(ptr nonnull %15)
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 28
  %17 = load i32, ptr %16, align 4
  %18 = icmp sgt i32 %17, -1
  br i1 %18, label %27, label %19

19:                                               ; preds = %14
  %20 = add nsw i32 %17, 16
  store i32 %20, ptr %16, align 4
  %21 = icmp samesign ult i32 %17, -15
  br i1 %21, label %22, label %27

22:                                               ; preds = %19
  %23 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %24 = load ptr, ptr %23, align 8
  %25 = sext i32 %17 to i64
  %26 = getelementptr inbounds i8, ptr %24, i64 %25
  br label %30

27:                                               ; preds = %19, %14
  %28 = load ptr, ptr %15, align 8
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 8
  store ptr %29, ptr %15, align 8
  br label %30

30:                                               ; preds = %27, %22
  %31 = phi ptr [ %26, %22 ], [ %28, %27 ]
  %32 = load double, ptr %31, align 8, !tbaa !10
  %33 = fcmp une double %32, 1.234000e+03
  br i1 %33, label %34, label %35

34:                                               ; preds = %30
  call void @abort() #5
  unreachable

35:                                               ; preds = %30
  call void @llvm.va_end.p0(ptr nonnull %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #4
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void (i32, i32, i32, i32, i32, i32, i32, fp128, ...) @test1(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, fp128 poison, i32 noundef 1234)
  tail call void (i32, i32, i32, i32, i32, i32, i32, fp128, i32, fp128, i32, fp128, i32, fp128, ...) @test2(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, fp128 poison, i32 poison, fp128 poison, i32 poison, fp128 poison, i32 poison, fp128 poison, i32 noundef 1234)
  tail call void (double, double, double, double, double, double, double, fp128, ...) @test3(double poison, double poison, double poison, double poison, double poison, double poison, double poison, fp128 poison, double noundef 1.234000e+03)
  tail call void (double, double, double, double, double, double, double, fp128, double, fp128, double, fp128, double, fp128, ...) @test4(double poison, double poison, double poison, double poison, double poison, double poison, double poison, fp128 poison, double poison, fp128 poison, double poison, fp128 poison, double poison, fp128 poison, double noundef 1.234000e+03)
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
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
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
