; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050121-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050121-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local { float, float } @foo_float(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 1
  %3 = sitofp i32 %2 to float
  %4 = add nsw i32 %0, -1
  %5 = sitofp i32 %4 to float
  %6 = insertvalue { float, float } poison, float %3, 0
  %7 = insertvalue { float, float } %6, float %5, 1
  ret { float, float } %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_float(ptr noundef writeonly captures(none) initializes((0, 4)) %0) local_unnamed_addr #1 {
  store float 6.000000e+00, ptr %0, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_float(ptr noundef writeonly captures(none) initializes((0, 4)) %0) local_unnamed_addr #1 {
  store float 4.000000e+00, ptr %0, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local { double, double } @foo_double(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 1
  %3 = sitofp i32 %2 to double
  %4 = add nsw i32 %0, -1
  %5 = sitofp i32 %4 to double
  %6 = insertvalue { double, double } poison, double %3, 0
  %7 = insertvalue { double, double } %6, double %5, 1
  ret { double, double } %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_double(ptr noundef writeonly captures(none) initializes((0, 8)) %0) local_unnamed_addr #1 {
  store double 6.000000e+00, ptr %0, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_double(ptr noundef writeonly captures(none) initializes((0, 8)) %0) local_unnamed_addr #1 {
  store double 4.000000e+00, ptr %0, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local { fp128, fp128 } @foo_ldouble_t(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 1
  %3 = sitofp i32 %2 to fp128
  %4 = add nsw i32 %0, -1
  %5 = sitofp i32 %4 to fp128
  %6 = insertvalue { fp128, fp128 } poison, fp128 %3, 0
  %7 = insertvalue { fp128, fp128 } %6, fp128 %5, 1
  ret { fp128, fp128 } %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_ldouble_t(ptr noundef writeonly captures(none) initializes((0, 16)) %0) local_unnamed_addr #1 {
  store fp128 0xL00000000000000004001800000000000, ptr %0, align 16, !tbaa !12
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_ldouble_t(ptr noundef writeonly captures(none) initializes((0, 16)) %0) local_unnamed_addr #1 {
  store fp128 0xL00000000000000004001000000000000, ptr %0, align 16, !tbaa !12
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i16 @foo_char(i32 noundef %0) local_unnamed_addr #0 {
  %2 = trunc i32 %0 to i8
  %3 = add i8 %2, 1
  %4 = add i8 %2, -1
  %5 = zext i8 %4 to i16
  %6 = shl nuw i16 %5, 8
  %7 = zext i8 %3 to i16
  %8 = or disjoint i16 %6, %7
  ret i16 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_char(ptr noundef writeonly captures(none) initializes((0, 1)) %0) local_unnamed_addr #1 {
  store i8 6, ptr %0, align 1, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_char(ptr noundef writeonly captures(none) initializes((0, 1)) %0) local_unnamed_addr #1 {
  store i8 4, ptr %0, align 1, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @foo_short(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add i32 %0, 1
  %3 = shl i32 %0, 16
  %4 = add i32 %3, -65536
  %5 = and i32 %2, 65535
  %6 = or disjoint i32 %4, %5
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_short(ptr noundef writeonly captures(none) initializes((0, 2)) %0) local_unnamed_addr #1 {
  store i16 6, ptr %0, align 2, !tbaa !15
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_short(ptr noundef writeonly captures(none) initializes((0, 2)) %0) local_unnamed_addr #1 {
  store i16 4, ptr %0, align 2, !tbaa !15
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i64 @foo_int(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 1
  %3 = add nsw i32 %0, -1
  %4 = zext i32 %3 to i64
  %5 = shl nuw i64 %4, 32
  %6 = zext i32 %2 to i64
  %7 = or disjoint i64 %5, %6
  ret i64 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_int(ptr noundef writeonly captures(none) initializes((0, 4)) %0) local_unnamed_addr #1 {
  store i32 6, ptr %0, align 4, !tbaa !17
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_int(ptr noundef writeonly captures(none) initializes((0, 4)) %0) local_unnamed_addr #1 {
  store i32 4, ptr %0, align 4, !tbaa !17
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @foo_long(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 1
  %3 = sext i32 %2 to i64
  %4 = add nsw i32 %0, -1
  %5 = sext i32 %4 to i64
  %6 = insertvalue [2 x i64] poison, i64 %3, 0
  %7 = insertvalue [2 x i64] %6, i64 %5, 1
  ret [2 x i64] %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_long(ptr noundef writeonly captures(none) initializes((0, 8)) %0) local_unnamed_addr #1 {
  store i64 6, ptr %0, align 8, !tbaa !19
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_long(ptr noundef writeonly captures(none) initializes((0, 8)) %0) local_unnamed_addr #1 {
  store i64 4, ptr %0, align 8, !tbaa !19
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @foo_llong(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 1
  %3 = sext i32 %2 to i64
  %4 = add nsw i32 %0, -1
  %5 = sext i32 %4 to i64
  %6 = insertvalue [2 x i64] poison, i64 %3, 0
  %7 = insertvalue [2 x i64] %6, i64 %5, 1
  ret [2 x i64] %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @bar_llong(ptr noundef writeonly captures(none) initializes((0, 8)) %0) local_unnamed_addr #1 {
  store i64 6, ptr %0, align 8, !tbaa !21
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @baz_llong(ptr noundef writeonly captures(none) initializes((0, 8)) %0) local_unnamed_addr #1 {
  store i64 4, ptr %0, align 8, !tbaa !21
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"long double", !8, i64 0}
!14 = !{!8, !8, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"short", !8, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !8, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"long", !8, i64 0}
!21 = !{!22, !22, i64 0}
!22 = !{!"long long", !8, i64 0}
