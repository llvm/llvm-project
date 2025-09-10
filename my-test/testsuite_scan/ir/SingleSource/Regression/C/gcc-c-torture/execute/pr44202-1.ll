; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44202-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr44202-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local range(i32 -2147483136, -2147483648) i32 @add512(i32 noundef %0, ptr noundef writeonly captures(none) %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %0, 512
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  store i32 %0, ptr %1, align 4, !tbaa !6
  br label %6

6:                                                ; preds = %5, %2
  ret i32 %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local range(i32 -2147483135, -2147483648) i32 @add513(i32 noundef %0, ptr noundef writeonly captures(none) %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %0, 513
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %2
  store i32 %0, ptr %1, align 4, !tbaa !6
  br label %6

6:                                                ; preds = %5, %2
  ret i32 %3
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store i32 -1, ptr %1, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  store i32 -1, ptr %2, align 4, !tbaa !6
  %3 = call i32 @add512(i32 noundef -512, ptr noundef nonnull %1)
  %4 = icmp ne i32 %3, 0
  %5 = load i32, ptr %1, align 4
  %6 = icmp ne i32 %5, -1
  %7 = select i1 %4, i1 true, i1 %6
  br i1 %7, label %14, label %8

8:                                                ; preds = %0
  %9 = call i32 @add513(i32 noundef -513, ptr noundef nonnull %2)
  %10 = icmp ne i32 %9, 0
  %11 = load i32, ptr %2, align 4
  %12 = icmp ne i32 %11, -513
  %13 = select i1 %10, i1 true, i1 %12
  br i1 %13, label %14, label %15

14:                                               ; preds = %8, %0
  tail call void @abort() #6
  unreachable

15:                                               ; preds = %8
  tail call void @exit(i32 noundef 0) #6
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
