; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr29695-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr29695-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 128, align 4
@b = dso_local local_unnamed_addr global i8 -128, align 4
@c = dso_local local_unnamed_addr global i64 2147483648, align 8
@d = dso_local local_unnamed_addr global i32 -2147483648, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 129) i32 @f1() local_unnamed_addr #0 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = and i32 %1, 128
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 129) i32 @f2() local_unnamed_addr #0 {
  %1 = load i8, ptr @b, align 4, !tbaa !10
  %2 = and i8 %1, -128
  %3 = zext i8 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 897) i32 @f3() local_unnamed_addr #0 {
  %1 = load i8, ptr @b, align 4, !tbaa !10
  %2 = icmp sgt i8 %1, -1
  %3 = select i1 %2, i32 0, i32 896
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 -128, 1) i32 @f4() local_unnamed_addr #0 {
  %1 = load i8, ptr @b, align 4, !tbaa !10
  %2 = icmp sgt i8 %1, -1
  %3 = select i1 %2, i32 0, i32 -128
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i64 0, 2147483649) i64 @f5() local_unnamed_addr #0 {
  %1 = load i64, ptr @c, align 8, !tbaa !11
  %2 = and i64 %1, 2147483648
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i64 0, 2147483649) i64 @f6() local_unnamed_addr #0 {
  %1 = load i32, ptr @d, align 4, !tbaa !6
  %2 = and i32 %1, -2147483648
  %3 = zext i32 %2 to i64
  ret i64 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i64 0, 15032385537) i64 @f7() local_unnamed_addr #0 {
  %1 = load i32, ptr @d, align 4, !tbaa !6
  %2 = icmp sgt i32 %1, -1
  %3 = select i1 %2, i64 0, i64 15032385536
  ret i64 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i64 -2147483648, 1) i64 @f8() local_unnamed_addr #0 {
  %1 = load i32, ptr @d, align 4, !tbaa !6
  %2 = icmp sgt i32 %1, -1
  %3 = select i1 %2, i64 0, i64 -2147483648
  ret i64 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!12 = !{!"long long", !8, i64 0}
