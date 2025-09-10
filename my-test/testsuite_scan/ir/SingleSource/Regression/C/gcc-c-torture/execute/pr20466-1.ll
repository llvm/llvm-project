; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20466-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20466-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local noundef i32 @f(ptr noundef captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef %2, ptr noundef writeonly captures(none) initializes((0, 8)) %3, ptr noundef readonly captures(none) %4) local_unnamed_addr #0 {
  %6 = load i32, ptr %1, align 4, !tbaa !6
  %7 = load ptr, ptr %0, align 8, !tbaa !10
  store i32 %6, ptr %7, align 4, !tbaa !6
  store ptr %2, ptr %0, align 8, !tbaa !10
  %8 = load ptr, ptr %4, align 8, !tbaa !10
  store ptr %8, ptr %3, align 8, !tbaa !10
  %9 = load ptr, ptr %0, align 8, !tbaa !10
  store i32 99, ptr %9, align 4, !tbaa !6
  ret i32 3
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store i32 42, ptr %1, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  store i32 66, ptr %2, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  store i32 1, ptr %3, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  store i32 55, ptr %4, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  store ptr %1, ptr %5, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #5
  store ptr %4, ptr %7, align 8, !tbaa !10
  %8 = call i32 @f(ptr noundef nonnull %5, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %6, ptr noundef nonnull %7)
  %9 = load i32, ptr %1, align 4, !tbaa !6
  %10 = icmp ne i32 %9, 66
  %11 = load ptr, ptr %5, align 8
  %12 = icmp ne ptr %11, %3
  %13 = select i1 %10, i1 true, i1 %12
  %14 = load i32, ptr %3, align 4
  %15 = icmp ne i32 %14, 99
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %23, label %17

17:                                               ; preds = %0
  %18 = load ptr, ptr %6, align 8, !tbaa !10
  %19 = icmp ne ptr %18, %4
  %20 = load i32, ptr %4, align 4
  %21 = icmp ne i32 %20, 55
  %22 = select i1 %19, i1 true, i1 %21
  br i1 %22, label %23, label %24

23:                                               ; preds = %17, %0
  call void @abort() #6
  unreachable

24:                                               ; preds = %17
  call void @exit(i32 noundef 0) #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

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
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
