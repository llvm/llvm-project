; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85756.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85756.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@h = dso_local local_unnamed_addr global i32 10, align 4
@p = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i16 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: nounwind uwtable
define dso_local void @bar(i32 noundef %0) local_unnamed_addr #0 {
  tail call void asm sideeffect "", "r,~{memory}"(i32 %0) #2, !srcloc !6
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  br label %2

2:                                                ; preds = %28, %0
  %3 = phi i32 [ 1, %0 ], [ %4, %28 ]
  %4 = phi i32 [ 430523, %0 ], [ %8, %28 ]
  %5 = phi i32 [ 1, %0 ], [ %17, %28 ]
  %6 = phi ptr [ @h, %0 ], [ %25, %28 ]
  store i32 %4, ptr @p, align 4, !tbaa !7
  %7 = or i32 %3, -65536
  %8 = sub i32 %5, %7
  %9 = load i16, ptr @b, align 4, !tbaa !11
  %10 = sext i16 %9 to i32
  %11 = sub i32 7, %5
  %12 = shl i32 %10, %11
  store i32 %12, ptr @f, align 4, !tbaa !7
  %13 = icmp ne i32 %8, 0
  %14 = icmp ne i16 %9, 0
  %15 = select i1 %13, i1 true, i1 %14
  %16 = zext i1 %15 to i32
  %17 = xor i32 %12, %16
  %18 = icmp ult i32 %4, %8
  br i1 %18, label %19, label %23

19:                                               ; preds = %2
  %20 = load i32, ptr %1, align 4, !tbaa !7
  %21 = icmp slt i32 %20, 3
  %22 = zext i1 %21 to i32
  store i32 %22, ptr %6, align 4, !tbaa !7
  br label %23

23:                                               ; preds = %19, %2
  %24 = icmp eq i32 %12, %16
  %25 = select i1 %24, ptr %1, ptr %6
  %26 = load i32, ptr @c, align 4, !tbaa !7
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %30, label %28

28:                                               ; preds = %23
  %29 = load i32, ptr @a, align 4, !tbaa !7
  tail call void asm sideeffect "", "r,~{memory}"(i32 %29) #2, !srcloc !6
  br label %2

30:                                               ; preds = %23
  %31 = load i32, ptr %25, align 4, !tbaa !7
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %33, label %35

33:                                               ; preds = %30
  %34 = load ptr, ptr @e, align 8, !tbaa !13
  store i32 1, ptr %34, align 4, !tbaa !7
  br label %35

35:                                               ; preds = %33, %30
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  br label %2

2:                                                ; preds = %28, %0
  %3 = phi i32 [ 1, %0 ], [ %4, %28 ]
  %4 = phi i32 [ 430523, %0 ], [ %8, %28 ]
  %5 = phi i32 [ 1, %0 ], [ %17, %28 ]
  %6 = phi ptr [ @h, %0 ], [ %25, %28 ]
  store i32 %4, ptr @p, align 4, !tbaa !7
  %7 = or i32 %3, -65536
  %8 = sub i32 %5, %7
  %9 = load i16, ptr @b, align 4, !tbaa !11
  %10 = sext i16 %9 to i32
  %11 = sub i32 7, %5
  %12 = shl i32 %10, %11
  store i32 %12, ptr @f, align 4, !tbaa !7
  %13 = icmp ne i32 %8, 0
  %14 = icmp ne i16 %9, 0
  %15 = select i1 %13, i1 true, i1 %14
  %16 = zext i1 %15 to i32
  %17 = xor i32 %12, %16
  %18 = icmp ult i32 %4, %8
  br i1 %18, label %19, label %23

19:                                               ; preds = %2
  %20 = load i32, ptr %1, align 4, !tbaa !7
  %21 = icmp slt i32 %20, 3
  %22 = zext i1 %21 to i32
  store i32 %22, ptr %6, align 4, !tbaa !7
  br label %23

23:                                               ; preds = %19, %2
  %24 = icmp eq i32 %12, %16
  %25 = select i1 %24, ptr %1, ptr %6
  %26 = load i32, ptr @c, align 4, !tbaa !7
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %30, label %28

28:                                               ; preds = %23
  %29 = load i32, ptr @a, align 4, !tbaa !7
  tail call void asm sideeffect "", "r,~{memory}"(i32 %29) #2, !srcloc !6
  br label %2

30:                                               ; preds = %23
  %31 = load i32, ptr %25, align 4, !tbaa !7
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %33, label %35

33:                                               ; preds = %30
  %34 = load ptr, ptr @e, align 8, !tbaa !13
  store i32 1, ptr %34, align 4, !tbaa !7
  br label %35

35:                                               ; preds = %30, %33
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  ret i32 0
}

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 204}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"short", !9, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 int", !15, i64 0}
!15 = !{!"any pointer", !9, i64 0}
