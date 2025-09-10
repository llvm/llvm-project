; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33992.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33992.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @bar(i64 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i64 %0, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %1
  tail call void @abort() #4
  unreachable

4:                                                ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @do_test(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i64, ptr %0, align 8, !tbaa !6
  br label %3

3:                                                ; preds = %3, %1
  %4 = phi i64 [ %8, %3 ], [ 0, %1 ]
  %5 = lshr exact i64 -9223372036854775808, %4
  %6 = and i64 %5, %2
  %7 = icmp eq i64 %6, 0
  %8 = add nuw nsw i64 %4, 1
  br i1 %7, label %3, label %9

9:                                                ; preds = %3
  %10 = and i64 %4, 4294967295
  tail call void @bar(i64 noundef %10)
  %11 = load i64, ptr %0, align 8, !tbaa !6
  br label %12

12:                                               ; preds = %12, %9
  %13 = phi i64 [ %17, %12 ], [ 0, %9 ]
  %14 = lshr exact i64 -9223372036854775808, %13
  %15 = and i64 %14, %11
  %16 = icmp eq i64 %15, 0
  %17 = add nuw nsw i64 %13, 1
  br i1 %16, label %12, label %18

18:                                               ; preds = %12
  %19 = and i64 %13, 4294967295
  tail call void @bar(i64 noundef %19)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca i64, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store i64 -9223372036854775807, ptr %1, align 8, !tbaa !6
  call void @do_test(ptr noundef nonnull %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
