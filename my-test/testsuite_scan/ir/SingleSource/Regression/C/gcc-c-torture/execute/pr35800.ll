; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr35800.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr35800.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"int\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"short\00", align 1
@.str.33 = private unnamed_addr constant [10 x i8] c"integer*8\00", align 1
@switch.table.stab_xcoff_builtin_type = private unnamed_addr constant [34 x ptr] [ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.33, ptr @.str.2, ptr @.str.2, ptr @.str], align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 256) i32 @stab_xcoff_builtin_type(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, -34
  br i1 %2, label %10, label %3

3:                                                ; preds = %1
  %4 = sext i32 %0 to i64
  %5 = getelementptr ptr, ptr @switch.table.stab_xcoff_builtin_type, i64 %4
  %6 = getelementptr i8, ptr %5, i64 272
  %7 = load ptr, ptr %6, align 8
  %8 = load i8, ptr %7, align 1, !tbaa !6
  %9 = zext i8 %8 to i32
  br label %10

10:                                               ; preds = %1, %3
  %11 = phi i32 [ %9, %3 ], [ 0, %1 ]
  ret i32 %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
