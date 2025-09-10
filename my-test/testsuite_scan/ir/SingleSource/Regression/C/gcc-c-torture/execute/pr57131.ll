; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57131.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57131.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca i32, align 4
  %2 = alloca i64, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i64, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store volatile i32 0, ptr %1, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  store volatile i64 0, ptr %2, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  store volatile i32 0, ptr %3, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  store volatile i32 1, ptr %4, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  store volatile i32 1, ptr %5, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store volatile i64 1, ptr %6, align 8, !tbaa !10
  %7 = load volatile i32, ptr %1, align 4, !tbaa !6
  %8 = sext i32 %7 to i64
  %9 = load volatile i64, ptr %2, align 8, !tbaa !10
  %10 = load volatile i32, ptr %3, align 4, !tbaa !6
  %11 = zext nneg i32 %10 to i64
  %12 = shl i64 %9, %11
  %13 = mul nsw i64 %12, %8
  %14 = load volatile i32, ptr %4, align 4, !tbaa !6
  %15 = load volatile i32, ptr %5, align 4, !tbaa !6
  %16 = mul nsw i32 %15, %14
  %17 = sext i32 %16 to i64
  %18 = sdiv i64 %13, %17
  %19 = load volatile i64, ptr %6, align 8, !tbaa !10
  %20 = add nsw i64 %19, %18
  %21 = icmp eq i64 %20, 1
  br i1 %21, label %23, label %22

22:                                               ; preds = %0
  tail call void @abort() #3
  unreachable

23:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
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
!10 = !{!11, !11, i64 0}
!11 = !{!"long long", !8, i64 0}
