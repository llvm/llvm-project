; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050604-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050604-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.anon = type { <4 x i16> }
%union.anon.0 = type { <4 x float> }

@u = dso_local local_unnamed_addr global %union.anon zeroinitializer, align 8
@v = dso_local local_unnamed_addr global %union.anon.0 zeroinitializer, align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = load <4 x i16>, ptr @u, align 8, !tbaa !6
  %2 = add <4 x i16> %1, <i16 24, i16 0, i16 0, i16 0>
  store <4 x i16> %2, ptr @u, align 8, !tbaa !6
  %3 = load <4 x float>, ptr @v, align 16, !tbaa !6
  %4 = fadd <4 x float> %3, <float 1.800000e+01, float 2.000000e+01, float 2.200000e+01, float 0.000000e+00>
  %5 = fadd <4 x float> %4, <float 1.800000e+01, float 2.000000e+01, float 2.200000e+01, float 0.000000e+00>
  store <4 x float> %5, ptr @v, align 16, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load <4 x i16>, ptr @u, align 8, !tbaa !6
  %2 = add <4 x i16> %1, <i16 24, i16 0, i16 0, i16 0>
  store <4 x i16> %2, ptr @u, align 8, !tbaa !6
  %3 = load <4 x float>, ptr @v, align 16, !tbaa !6
  %4 = freeze <4 x float> %3
  %5 = fadd <4 x float> %4, <float 1.800000e+01, float 2.000000e+01, float 2.200000e+01, float 0.000000e+00>
  %6 = fadd <4 x float> %5, <float 1.800000e+01, float 2.000000e+01, float 2.200000e+01, float 0.000000e+00>
  store <4 x float> %6, ptr @v, align 16, !tbaa !6
  %7 = bitcast <4 x i16> %2 to i64
  %8 = and i64 %7, 65535
  %9 = icmp ne i64 %8, 24
  %10 = shufflevector <4 x i16> %2, <4 x i16> poison, <4 x i32> <i32 poison, i32 2, i32 poison, i32 poison>
  %11 = shufflevector <4 x i16> %2, <4 x i16> poison, <4 x i32> <i32 poison, i32 3, i32 poison, i32 poison>
  %12 = or <4 x i16> %10, %11
  %13 = or <4 x i16> %12, %2
  %14 = extractelement <4 x i16> %13, i64 1
  %15 = icmp ne i16 %14, 0
  %16 = or i1 %9, %15
  br i1 %16, label %17, label %18

17:                                               ; preds = %0
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %0
  %19 = fcmp une <4 x float> %6, <float 3.600000e+01, float 4.000000e+01, float 4.400000e+01, float 0.000000e+00>
  %20 = bitcast <4 x i1> %19 to i4
  %21 = icmp eq i4 %20, 0
  br i1 %21, label %23, label %22

22:                                               ; preds = %18
  tail call void @abort() #3
  unreachable

23:                                               ; preds = %18
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
