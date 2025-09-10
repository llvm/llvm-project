; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/ConditionalExpr.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/ConditionalExpr.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.T = type { i32 }

@_ZL1X = internal unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [14 x i8] c"Construct %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [13 x i8] c"Destruct %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [22 x i8] c"Overwrite %d with %d\0A\00", align 1

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z4funcRK1T(ptr dead_on_unwind noalias writable writeonly sret(%struct.T) align 4 captures(none) initializes((0, 4)) %0, ptr noundef nonnull readnone align 4 captures(none) dereferenceable(4) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @_ZL1X, align 4, !tbaa !6
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @_ZL1X, align 4, !tbaa !6
  store i32 %4, ptr %0, align 4, !tbaa !10
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %4)
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z4testb(ptr dead_on_unwind noalias writable writeonly sret(%struct.T) align 4 captures(none) initializes((0, 4)) %0, i1 noundef %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = load i32, ptr @_ZL1X, align 4, !tbaa !6
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @_ZL1X, align 4, !tbaa !6
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %4)
  br i1 %1, label %11, label %6

6:                                                ; preds = %2
  %7 = load i32, ptr @_ZL1X, align 4, !tbaa !6, !noalias !12
  %8 = add nsw i32 %7, 1
  store i32 %8, ptr @_ZL1X, align 4, !tbaa !6, !noalias !12
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %8), !noalias !12
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %4)
  br label %11

11:                                               ; preds = %2, %6
  %12 = phi i32 [ %8, %6 ], [ %4, %2 ]
  store i32 %12, ptr %0, align 4, !tbaa !10
  ret void
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nofree norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 personality ptr @__gxx_personality_v0 {
  %1 = load i32, ptr @_ZL1X, align 4, !tbaa !6
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @_ZL1X, align 4, !tbaa !6
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %2)
  %4 = load i32, ptr @_ZL1X, align 4, !tbaa !6, !noalias !15
  %5 = add nsw i32 %4, 1
  store i32 %5, ptr @_ZL1X, align 4, !tbaa !6, !noalias !15
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %5), !noalias !15
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %2, i32 noundef %5)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %5)
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %5)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !7, i64 0}
!11 = !{!"_ZTS1T", !7, i64 0}
!12 = !{!13}
!13 = distinct !{!13, !14, !"_Z4funcRK1T: argument 0"}
!14 = distinct !{!14, !"_Z4funcRK1T"}
!15 = !{!16}
!16 = distinct !{!16, !17, !"_Z4testb: argument 0"}
!17 = distinct !{!17, !"_Z4testb"}
