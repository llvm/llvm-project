; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2009-12-07-StructReturn.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2009-12-07-StructReturn.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.barstruct = type { %struct.foostruct, float }
%struct.foostruct = type { [3 x float] }

@bar_ary = dso_local global [4 x %struct.barstruct] zeroinitializer, align 4
@str = private unnamed_addr constant [50 x i8] c"error: store clobbered memory outside destination\00", align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef nonnull ptr @spooky(i32 noundef %0) local_unnamed_addr #0 {
  %2 = sext i32 %0 to i64
  %3 = getelementptr inbounds %struct.barstruct, ptr @bar_ary, i64 %2, i32 1
  store float 1.420000e+02, ptr %3, align 4, !tbaa !6
  ret ptr %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local %struct.foostruct @foobify() local_unnamed_addr #1 {
  ret %struct.foostruct { [3 x float] [float 4.200000e+01, float 4.200000e+01, float 4.200000e+01] }
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #2 {
  %3 = tail call ptr @spooky(i32 noundef 0)
  %4 = tail call %struct.foostruct @foobify()
  %5 = extractvalue %struct.foostruct %4, 0
  %6 = extractvalue [3 x float] %5, 0
  %7 = extractvalue [3 x float] %5, 1
  %8 = extractvalue [3 x float] %5, 2
  store float %6, ptr @bar_ary, align 4
  store float %7, ptr getelementptr inbounds nuw (i8, ptr @bar_ary, i64 4), align 4
  store float %8, ptr getelementptr inbounds nuw (i8, ptr @bar_ary, i64 8), align 4, !tbaa !12
  %9 = load float, ptr %3, align 4, !tbaa !13
  %10 = fcmp une float %9, 1.420000e+02
  br i1 %10, label %11, label %13

11:                                               ; preds = %2
  %12 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @abort() #5
  unreachable

13:                                               ; preds = %2
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !11, i64 12}
!7 = !{!"", !8, i64 0, !11, i64 12}
!8 = !{!"", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"float", !9, i64 0}
!12 = !{!9, !9, i64 0}
!13 = !{!11, !11, i64 0}
