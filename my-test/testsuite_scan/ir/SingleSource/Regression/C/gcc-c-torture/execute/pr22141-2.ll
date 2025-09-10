; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr22141-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr22141-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { %struct.T }
%struct.T = type { i8, i8, i8, i8 }
%struct.U = type { [4 x %struct.S] }

@u = dso_local global %struct.S zeroinitializer, align 16

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @c1(ptr noundef captures(none) %0) local_unnamed_addr #0 {
  %2 = load i8, ptr %0, align 1, !tbaa !6
  %3 = icmp eq i8 %2, 1
  br i1 %3, label %4, label %16

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %6 = load i8, ptr %5, align 1, !tbaa !10
  %7 = icmp eq i8 %6, 2
  br i1 %7, label %8, label %16

8:                                                ; preds = %4
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %10 = load i8, ptr %9, align 1, !tbaa !11
  %11 = icmp eq i8 %10, 3
  br i1 %11, label %12, label %16

12:                                               ; preds = %8
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 3
  %14 = load i8, ptr %13, align 1, !tbaa !12
  %15 = icmp eq i8 %14, 4
  br i1 %15, label %17, label %16

16:                                               ; preds = %12, %8, %4, %1
  tail call void @abort() #6
  unreachable

17:                                               ; preds = %12
  store i32 -1431655766, ptr %0, align 1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @c2(ptr noundef captures(none) %0) local_unnamed_addr #0 {
  tail call void @c1(ptr noundef %0)
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @c3(ptr noundef captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  tail call void @c2(ptr noundef nonnull %2)
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f1() local_unnamed_addr #2 {
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr @u, align 16, !tbaa !13
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f2() local_unnamed_addr #2 {
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr @u, align 16, !tbaa !13
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f3() local_unnamed_addr #2 {
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr @u, align 16, !tbaa !13
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @f4() local_unnamed_addr #0 {
  %1 = alloca %struct.S, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr %1, align 16, !tbaa !13
  call void @c2(ptr noundef nonnull %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @f5(ptr noundef writeonly captures(none) initializes((0, 4)) %0) local_unnamed_addr #4 {
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr %0, align 1, !tbaa !13
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @f6() local_unnamed_addr #0 {
  %1 = alloca %struct.U, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr %2, align 8, !tbaa !13
  call void @c3(ptr noundef nonnull %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @f7(ptr noundef writeonly captures(none) initializes((8, 12)) %0) local_unnamed_addr #4 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr %2, align 1, !tbaa !13
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = alloca %struct.U, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  tail call void @f1()
  tail call void @c2(ptr noundef nonnull @u)
  tail call void @f2()
  tail call void @c1(ptr noundef nonnull @u)
  tail call void @f3()
  tail call void @c2(ptr noundef nonnull @u)
  tail call void @f4()
  tail call void @f5(ptr noundef nonnull @u)
  tail call void @c2(ptr noundef nonnull @u)
  tail call void @f6()
  call void @f7(ptr noundef nonnull %1)
  call void @c3(ptr noundef nonnull %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"T", !8, i64 0, !8, i64 1, !8, i64 2, !8, i64 3}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!7, !8, i64 1}
!11 = !{!7, !8, i64 2}
!12 = !{!7, !8, i64 3}
!13 = !{!8, !8, i64 0}
