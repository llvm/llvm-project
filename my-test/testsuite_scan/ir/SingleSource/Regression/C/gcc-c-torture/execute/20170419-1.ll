; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20170419-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20170419-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@x = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store volatile i32 0, ptr %1, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  store volatile i32 -2147483647, ptr %2, align 4, !tbaa !6
  %3 = load volatile i32, ptr %1, align 4, !tbaa !6
  %4 = load volatile i32, ptr %2, align 4, !tbaa !6
  %5 = load volatile i32, ptr %1, align 4, !tbaa !6
  %6 = load volatile i32, ptr %2, align 4, !tbaa !6
  %7 = load volatile i32, ptr %1, align 4, !tbaa !6
  %8 = load volatile i32, ptr %2, align 4, !tbaa !6
  %9 = load volatile i32, ptr %1, align 4, !tbaa !6
  %10 = load volatile i32, ptr %2, align 4, !tbaa !6
  %11 = load volatile i32, ptr %1, align 4, !tbaa !6
  %12 = load volatile i32, ptr %2, align 4, !tbaa !6
  %13 = load volatile i32, ptr %1, align 4, !tbaa !6
  %14 = load volatile i32, ptr %2, align 4, !tbaa !6
  %15 = load volatile i32, ptr %1, align 4, !tbaa !6
  %16 = load volatile i32, ptr %2, align 4, !tbaa !6
  %17 = load volatile i32, ptr %1, align 4, !tbaa !6
  %18 = load volatile i32, ptr %2, align 4, !tbaa !6
  %19 = load volatile i32, ptr %1, align 4, !tbaa !6
  %20 = load volatile i32, ptr %2, align 4, !tbaa !6
  %21 = load volatile i32, ptr %1, align 4, !tbaa !6
  %22 = load volatile i32, ptr %2, align 4, !tbaa !6
  %23 = load volatile i32, ptr %1, align 4, !tbaa !6
  %24 = load volatile i32, ptr %2, align 4, !tbaa !6
  %25 = load volatile i32, ptr %1, align 4, !tbaa !6
  %26 = load volatile i32, ptr %2, align 4, !tbaa !6
  %27 = load volatile i32, ptr %1, align 4, !tbaa !6
  %28 = load volatile i32, ptr %2, align 4, !tbaa !6
  %29 = load volatile i32, ptr %1, align 4, !tbaa !6
  %30 = load volatile i32, ptr %2, align 4, !tbaa !6
  %31 = load volatile i32, ptr %1, align 4, !tbaa !6
  %32 = load volatile i32, ptr %2, align 4, !tbaa !6
  %33 = load volatile i32, ptr %1, align 4, !tbaa !6
  %34 = load volatile i32, ptr %2, align 4, !tbaa !6
  %35 = load volatile i32, ptr %1, align 4, !tbaa !6
  %36 = load volatile i32, ptr %2, align 4, !tbaa !6
  %37 = load volatile i32, ptr %1, align 4, !tbaa !6
  %38 = load volatile i32, ptr %2, align 4, !tbaa !6
  %39 = icmp eq i32 %37, 0
  %40 = zext i1 %39 to i32
  %41 = xor i32 %38, %40
  %42 = icmp ne i32 %41, -2147483648
  %43 = zext i1 %42 to i32
  store i32 %43, ptr @x, align 4, !tbaa !6
  br i1 %42, label %44, label %45

44:                                               ; preds = %0
  tail call void @abort() #3
  unreachable

45:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
