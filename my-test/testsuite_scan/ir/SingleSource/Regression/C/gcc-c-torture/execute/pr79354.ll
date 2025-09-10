; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr79354.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr79354.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@g = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i64 0, align 8
@e = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@b = dso_local global i32 0, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define dso_local void @foo(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @f, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  store i32 0, ptr @g, align 4, !tbaa !6
  br i1 %3, label %8, label %4

4:                                                ; preds = %1
  %5 = load i64, ptr @d, align 8
  %6 = uitofp i64 %5 to float
  store i32 31, ptr @g, align 4, !tbaa !6
  store float %6, ptr @b, align 4
  %7 = load i32, ptr %0, align 4, !tbaa !6
  store i32 %7, ptr @b, align 4, !tbaa !6
  store i32 32, ptr @g, align 4, !tbaa !6
  store float %6, ptr @e, align 4, !tbaa !10
  br label %9

8:                                                ; preds = %1
  store i32 32, ptr @g, align 4, !tbaa !6
  br label %9

9:                                                ; preds = %8, %4
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #3
  store i32 5, ptr %1, align 4, !tbaa !6
  store i32 1, ptr @f, align 4, !tbaa !6
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !12
  call void @foo(ptr noundef nonnull %1)
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !13
  tail call void @foo(ptr noundef nonnull @b)
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #3
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !8, i64 0}
!12 = !{i64 311}
!13 = !{i64 359}
!14 = !{i64 407}
