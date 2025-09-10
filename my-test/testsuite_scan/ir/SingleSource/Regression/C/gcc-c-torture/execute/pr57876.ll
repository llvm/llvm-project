; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57876.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57876.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i32 1, align 4
@c = dso_local global i32 0, align 4
@d = dso_local local_unnamed_addr global ptr @c, align 8
@f = dso_local local_unnamed_addr global i32 0, align 4
@j = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca i32, align 4
  %2 = load i32, ptr @b, align 4, !tbaa !6
  %3 = load ptr, ptr @d, align 8, !tbaa !10
  %4 = load i32, ptr @a, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #3
  store i32 0, ptr @f, align 4, !tbaa !6
  %5 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %5, ptr @j, align 4, !tbaa !6
  %6 = mul nsw i32 %4, %5
  %7 = add nsw i32 %6, -1
  store i32 %7, ptr @h, align 4, !tbaa !6
  store i32 1, ptr @f, align 4, !tbaa !6
  %8 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %8, ptr @j, align 4, !tbaa !6
  %9 = mul nsw i32 %4, %8
  %10 = add nsw i32 %9, -1
  store i32 %10, ptr @h, align 4, !tbaa !6
  store i32 2, ptr @f, align 4, !tbaa !6
  %11 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %11, ptr @j, align 4, !tbaa !6
  %12 = mul nsw i32 %4, %11
  %13 = add nsw i32 %12, -1
  store i32 %13, ptr @h, align 4, !tbaa !6
  store i32 3, ptr @f, align 4, !tbaa !6
  %14 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %14, ptr @j, align 4, !tbaa !6
  %15 = mul nsw i32 %4, %14
  %16 = add nsw i32 %15, -1
  store i32 %16, ptr @h, align 4, !tbaa !6
  store i32 4, ptr @f, align 4, !tbaa !6
  %17 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %17, ptr @j, align 4, !tbaa !6
  %18 = mul nsw i32 %4, %17
  %19 = add nsw i32 %18, -1
  store i32 %19, ptr @h, align 4, !tbaa !6
  store i32 5, ptr @f, align 4, !tbaa !6
  %20 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %20, ptr @j, align 4, !tbaa !6
  %21 = mul nsw i32 %4, %20
  %22 = add nsw i32 %21, -1
  store i32 %22, ptr @h, align 4, !tbaa !6
  store i32 6, ptr @f, align 4, !tbaa !6
  %23 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %23, ptr @j, align 4, !tbaa !6
  %24 = mul nsw i32 %4, %23
  %25 = add nsw i32 %24, -1
  store i32 %25, ptr @h, align 4, !tbaa !6
  store i32 7, ptr @f, align 4, !tbaa !6
  %26 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %26, ptr @j, align 4, !tbaa !6
  %27 = mul nsw i32 %4, %26
  %28 = add nsw i32 %27, -1
  store i32 %28, ptr @h, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #3
  store i32 0, ptr @f, align 4, !tbaa !6
  %29 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %29, ptr @j, align 4, !tbaa !6
  %30 = mul nsw i32 %4, %29
  %31 = add nsw i32 %30, -1
  store i32 %31, ptr @h, align 4, !tbaa !6
  store i32 1, ptr @f, align 4, !tbaa !6
  %32 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %32, ptr @j, align 4, !tbaa !6
  %33 = mul nsw i32 %4, %32
  %34 = add nsw i32 %33, -1
  store i32 %34, ptr @h, align 4, !tbaa !6
  store i32 2, ptr @f, align 4, !tbaa !6
  %35 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %35, ptr @j, align 4, !tbaa !6
  %36 = mul nsw i32 %4, %35
  %37 = add nsw i32 %36, -1
  store i32 %37, ptr @h, align 4, !tbaa !6
  store i32 3, ptr @f, align 4, !tbaa !6
  %38 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %38, ptr @j, align 4, !tbaa !6
  %39 = mul nsw i32 %4, %38
  %40 = add nsw i32 %39, -1
  store i32 %40, ptr @h, align 4, !tbaa !6
  store i32 4, ptr @f, align 4, !tbaa !6
  %41 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %41, ptr @j, align 4, !tbaa !6
  %42 = mul nsw i32 %4, %41
  %43 = add nsw i32 %42, -1
  store i32 %43, ptr @h, align 4, !tbaa !6
  store i32 5, ptr @f, align 4, !tbaa !6
  %44 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %44, ptr @j, align 4, !tbaa !6
  %45 = mul nsw i32 %4, %44
  %46 = add nsw i32 %45, -1
  store i32 %46, ptr @h, align 4, !tbaa !6
  store i32 6, ptr @f, align 4, !tbaa !6
  %47 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %47, ptr @j, align 4, !tbaa !6
  %48 = mul nsw i32 %4, %47
  %49 = add nsw i32 %48, -1
  store i32 %49, ptr @h, align 4, !tbaa !6
  store i32 7, ptr @f, align 4, !tbaa !6
  %50 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %50, ptr @j, align 4, !tbaa !6
  %51 = mul nsw i32 %4, %50
  %52 = add nsw i32 %51, -1
  store i32 %52, ptr @h, align 4, !tbaa !6
  store i32 8, ptr @f, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #3
  %53 = sext i32 %2 to i64
  %54 = icmp eq i32 %52, 0
  %55 = zext i1 %54 to i64
  %56 = icmp slt i64 %55, %53
  store ptr %1, ptr @g, align 8, !tbaa !10
  br i1 %56, label %58, label %57

57:                                               ; preds = %0
  call void @abort() #4
  unreachable

58:                                               ; preds = %0
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
