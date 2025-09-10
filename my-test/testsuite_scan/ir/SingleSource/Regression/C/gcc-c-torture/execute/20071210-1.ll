; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071210-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071210-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@bar.l = internal global [5 x ptr] [ptr blockaddress(@bar, %28), ptr blockaddress(@bar, %28), ptr blockaddress(@bar, %11), ptr blockaddress(@bar, %23), ptr blockaddress(@bar, %26)], align 8
@__const.main.s = private unnamed_addr constant [6 x i32] [i32 7, i32 8, i32 9, i32 10, i32 11, i32 12], align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local [2 x i64] @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp ne i32 %0, 10
  %5 = icmp ne i32 %1, 9
  %6 = or i1 %4, %5
  %7 = icmp ne i32 %2, 8
  %8 = or i1 %6, %7
  br i1 %8, label %9, label %10

9:                                                ; preds = %3
  tail call void @abort() #5
  unreachable

10:                                               ; preds = %3
  ret [2 x i64] [i64 8589934593, i64 17179869187]
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef ptr @bar(ptr noundef readonly captures(address_is_null) %0, ptr noundef captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp eq ptr %0, null
  br i1 %3, label %26, label %4

4:                                                ; preds = %2
  %5 = load ptr, ptr %0, align 8, !tbaa !6
  br label %6

6:                                                ; preds = %20, %4
  %7 = phi ptr [ %5, %4 ], [ %21, %20 ]
  %8 = phi ptr [ %1, %4 ], [ %22, %20 ]
  %9 = phi ptr [ %0, %4 ], [ %10, %20 ]
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 8
  br label %28

11:                                               ; preds = %28
  %12 = load ptr, ptr %10, align 8, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %14 = load i32, ptr %13, align 4, !tbaa !10
  %15 = getelementptr inbounds nuw i8, ptr %8, i64 4
  %16 = load i32, ptr %15, align 4, !tbaa !10
  %17 = load i32, ptr %8, align 4, !tbaa !10
  %18 = getelementptr inbounds i8, ptr %8, i64 -4
  %19 = tail call [2 x i64] @foo(i32 noundef %14, i32 noundef %16, i32 noundef %17)
  store <4 x i32> <i32 4, i32 3, i32 2, i32 1>, ptr %18, align 4, !tbaa !10
  br label %20

20:                                               ; preds = %11, %23
  %21 = phi ptr [ %24, %23 ], [ %12, %11 ]
  %22 = phi ptr [ %25, %23 ], [ %18, %11 ]
  br label %6

23:                                               ; preds = %28
  %24 = load ptr, ptr %10, align 8, !tbaa !6
  %25 = getelementptr inbounds nuw i8, ptr %8, i64 4
  store i32 23, ptr %25, align 4, !tbaa !10
  br label %20

26:                                               ; preds = %28, %2
  %27 = phi ptr [ @bar.l, %2 ], [ null, %28 ]
  ret ptr %27

28:                                               ; preds = %6, %28
  indirectbr ptr %7, [label %28, label %26, label %11, label %23]
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca [2 x ptr], align 8
  %2 = alloca [6 x i32], align 16
  %3 = tail call ptr @bar(ptr noundef null, ptr noundef null)
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load ptr, ptr %4, align 8, !tbaa !6
  store ptr %5, ptr %1, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !tbaa !6
  store ptr %8, ptr %6, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(24) %2, ptr noundef nonnull align 4 dereferenceable(24) @__const.main.s, i64 24, i1 false)
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %10 = call ptr @bar(ptr noundef nonnull %1, ptr noundef nonnull %9)
  %11 = icmp ne ptr %10, null
  %12 = load <4 x i32>, ptr %2, align 16
  %13 = freeze <4 x i32> %12
  %14 = icmp ne <4 x i32> %13, <i32 4, i32 3, i32 2, i32 1>
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %16 = load i32, ptr %15, align 16
  %17 = freeze i32 %16
  %18 = icmp ne i32 %17, 11
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 20
  %20 = load i32, ptr %19, align 4
  %21 = icmp ne i32 %20, 12
  %22 = bitcast <4 x i1> %14 to i4
  %23 = icmp ne i4 %22, 0
  %24 = or i1 %23, %18
  %25 = or i1 %24, %11
  %26 = select i1 %25, i1 true, i1 %21
  br i1 %26, label %27, label %28

27:                                               ; preds = %0
  call void @abort() #5
  unreachable

28:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
