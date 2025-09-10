; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/byval-alignment.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/byval-alignment.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.s0 = type { fp128, fp128 }

@g0 = dso_local local_unnamed_addr global %struct.s0 zeroinitializer, align 16
@.str = private unnamed_addr constant [24 x i8] c"g0.x: %.4f, g0.y: %.4f\0A\00", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f0(i32 noundef %0, [2 x fp128] alignstack(16) %1) local_unnamed_addr #0 {
  %3 = extractvalue [2 x fp128] %1, 0
  %4 = extractvalue [2 x fp128] %1, 1
  %5 = sitofp i32 %0 to fp128
  %6 = fadd fp128 %3, %5
  store fp128 %6, ptr @g0, align 16, !tbaa !6
  %7 = fadd fp128 %4, %5
  store fp128 %7, ptr getelementptr inbounds nuw (i8, ptr @g0, i64 16), align 16, !tbaa !11
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @f0(i32 noundef 1, [2 x fp128] alignstack(16) [fp128 0xL00000000000000003FFF000000000000, fp128 0xL00000000000000004000000000000000])
  %1 = load fp128, ptr @g0, align 16, !tbaa !6
  %2 = fptrunc fp128 %1 to double
  %3 = load fp128, ptr getelementptr inbounds nuw (i8, ptr @g0, i64 16), align 16, !tbaa !11
  %4 = fptrunc fp128 %3 to double
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %2, double noundef %4)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"s0", !8, i64 0, !8, i64 16}
!8 = !{!"long double", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 16}
