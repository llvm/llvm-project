; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr67781.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr67781.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@s = dso_local local_unnamed_addr global { i32, i8, [3 x i8] } { i32 1193046, i8 120, [3 x i8] zeroinitializer }, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local i32 @pr67781() local_unnamed_addr #0 {
  %1 = load i32, ptr @s, align 4, !tbaa !6
  %2 = shl i32 %1, 8
  %3 = load i8, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !11
  %4 = zext i8 %3 to i32
  %5 = or disjoint i32 %2, %4
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @s, align 4, !tbaa !6
  %2 = shl i32 %1, 8
  %3 = load i8, ptr getelementptr inbounds nuw (i8, ptr @s, i64 4), align 4, !tbaa !11
  %4 = zext i8 %3 to i32
  %5 = or disjoint i32 %2, %4
  %6 = icmp eq i32 %5, 305419896
  br i1 %6, label %8, label %7

7:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !8, i64 0}
!7 = !{!"", !8, i64 0, !9, i64 4}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !9, i64 4}
