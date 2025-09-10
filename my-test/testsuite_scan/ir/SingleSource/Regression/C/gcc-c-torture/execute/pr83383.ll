; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr83383.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr83383.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i64 16, align 8
@b = dso_local local_unnamed_addr global i8 -61, align 4
@c = dso_local local_unnamed_addr global i64 -1, align 8
@d = dso_local local_unnamed_addr global i8 1, align 4
@e = dso_local local_unnamed_addr global [2 x i64] [i64 3625445792498952486, i64 0], align 8
@f = dso_local local_unnamed_addr global [2 x i64] [i64 0, i64 8985037393681294663], align 8
@g = dso_local local_unnamed_addr global i64 5052410635626804928, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = load i64, ptr @a, align 8, !tbaa !6
  %2 = trunc i64 %1 to i32
  %3 = shl i32 %2, 24
  %4 = ashr exact i32 %3, 24
  %5 = load i8, ptr @b, align 4, !tbaa !10
  %6 = zext i8 %5 to i32
  %7 = icmp slt i32 %4, %6
  %8 = zext i1 %7 to i64
  store i64 %8, ptr @a, align 8, !tbaa !6
  %9 = load i8, ptr @d, align 4, !tbaa !10
  %10 = icmp eq i8 %9, 0
  %11 = load i64, ptr @e, align 8
  %12 = select i1 %10, i64 0, i64 %11
  %13 = load i64, ptr getelementptr inbounds nuw (i8, ptr @f, i64 8), align 8, !tbaa !6
  %14 = icmp ne i64 %13, 0
  %15 = select i1 %7, i1 %14, i1 false
  %16 = load i64, ptr @g, align 8
  %17 = select i1 %15, i64 1, i64 %16
  %18 = sub i64 %12, %17
  store i64 %18, ptr @c, align 8, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i64, ptr @a, align 8, !tbaa !6
  %2 = trunc i64 %1 to i32
  %3 = shl i32 %2, 24
  %4 = ashr exact i32 %3, 24
  %5 = load i8, ptr @b, align 4, !tbaa !10
  %6 = zext i8 %5 to i32
  %7 = icmp slt i32 %4, %6
  %8 = zext i1 %7 to i64
  store i64 %8, ptr @a, align 8, !tbaa !6
  %9 = load i8, ptr @d, align 4, !tbaa !10
  %10 = icmp eq i8 %9, 0
  %11 = load i64, ptr @e, align 8
  %12 = select i1 %10, i64 0, i64 %11
  %13 = load i64, ptr getelementptr inbounds nuw (i8, ptr @f, i64 8), align 8, !tbaa !6
  %14 = icmp ne i64 %13, 0
  %15 = select i1 %7, i1 %14, i1 false
  %16 = load i64, ptr @g, align 8
  %17 = select i1 %15, i64 1, i64 %16
  %18 = sub i64 %12, %17
  store i64 %18, ptr @c, align 8, !tbaa !6
  %19 = icmp eq i64 %18, 3625445792498952485
  %20 = select i1 %7, i1 %19, i1 false
  br i1 %20, label %22, label %21

21:                                               ; preds = %0
  tail call void @abort() #3
  unreachable

22:                                               ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
