; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr66556.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr66556.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global { i32, i8, i8, i8, i8, i16, [2 x i8] } { i32 8355840, i8 6, i8 -128, i8 2, i8 0, i16 0, [2 x i8] zeroinitializer }, align 4
@g = dso_local local_unnamed_addr global { i32, i8, i8, i8, i8, i16, [2 x i8] } { i32 8, i8 0, i8 -128, i8 2, i8 0, i16 0, [2 x i8] zeroinitializer }, align 4
@h = dso_local local_unnamed_addr global i32 8, align 4
@e = dso_local global <{ i8, [236 x i8] }> <{ i8 4, [236 x i8] zeroinitializer }>, align 4
@d = dso_local global i16 0, align 2
@f = dso_local local_unnamed_addr global ptr @d, align 8
@i = dso_local local_unnamed_addr global [5 x i16] [i16 3, i16 0, i16 0, i16 0, i16 0], align 2
@c = dso_local global i32 0, align 4
@k = dso_local local_unnamed_addr global ptr @c, align 8
@a = dso_local local_unnamed_addr global i32 0, align 4
@j = dso_local local_unnamed_addr global i8 0, align 4
@l = dso_local local_unnamed_addr global i16 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @fn1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = sub i32 0, %0
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: none) uwtable
define dso_local void @fn2(i8 noundef %0) local_unnamed_addr #1 {
  %2 = zext i8 %0 to i32
  store i32 %2, ptr @a, align 4, !tbaa !6
  %3 = load volatile i8, ptr @e, align 4, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define dso_local i16 @fn3() local_unnamed_addr #2 {
  %1 = load ptr, ptr @k, align 8, !tbaa !11
  store i32 4, ptr %1, align 4, !tbaa !6
  %2 = load ptr, ptr @f, align 8, !tbaa !14
  %3 = load i16, ptr %2, align 2, !tbaa !16
  ret i16 %3
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = load i32, ptr @h, align 4, !tbaa !6
  %2 = icmp ne i32 %1, 0
  %3 = load i8, ptr @j, align 4
  %4 = icmp ne i8 %3, 0
  %5 = select i1 %2, i1 %4, i1 false
  %6 = zext i1 %5 to i32
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 4), align 4
  %8 = lshr i32 %7, 15
  %9 = and i32 %8, 7
  %10 = icmp samesign uge i32 %9, %6
  %11 = sext i1 %10 to i32
  %12 = load i32, ptr getelementptr inbounds nuw (i8, ptr @g, i64 4), align 4
  %13 = and i32 %12, 32767
  %14 = icmp ult i32 %13, %11
  %15 = zext i1 %14 to i16
  store i16 %15, ptr @l, align 4, !tbaa !16
  store i16 3, ptr getelementptr inbounds nuw (i8, ptr @i, i64 8), align 2, !tbaa !16
  %16 = load i32, ptr @b, align 4, !tbaa !18
  %17 = lshr i32 %16, 15
  %18 = and i32 %17, 255
  store i32 %18, ptr @a, align 4, !tbaa !6
  %19 = load volatile i8, ptr @e, align 4, !tbaa !10
  %20 = icmp eq i32 %18, 255
  br i1 %20, label %22, label %21

21:                                               ; preds = %0
  tail call void @abort() #5
  unreachable

22:                                               ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!10 = !{!8, !8, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"p1 int", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 short", !13, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"short", !8, i64 0}
!18 = !{!19, !7, i64 0}
!19 = !{!"", !7, i64 0, !7, i64 4, !7, i64 5, !17, i64 8}
