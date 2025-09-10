; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr27285.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr27285.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { i8, i8, i8, [16 x i8] }

@__const.main.x = private unnamed_addr constant { i8, i8, i8, <{ i8, i8, i8, i8, [12 x i8] }> } { i8 0, i8 25, i8 0, <{ i8, i8, i8, i8, [12 x i8] }> <{ i8 -86, i8 -69, i8 -52, i8 -35, [12 x i8] zeroinitializer }> }, align 1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @foo(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 1
  %4 = load i8, ptr %3, align 1, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 3
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 3
  %7 = icmp eq i8 %4, 0
  br i1 %7, label %26, label %8

8:                                                ; preds = %2
  %9 = zext i8 %4 to i32
  br label %10

10:                                               ; preds = %8, %10
  %11 = phi i64 [ 0, %8 ], [ %22, %10 ]
  %12 = phi i32 [ %9, %8 ], [ %24, %10 ]
  %13 = icmp samesign ugt i32 %12, 7
  %14 = sub nuw nsw i32 8, %12
  %15 = shl nuw nsw i32 255, %14
  %16 = trunc i32 %15 to i8
  %17 = select i1 %13, i8 -1, i8 %16
  %18 = getelementptr inbounds nuw i8, ptr %5, i64 %11
  %19 = load i8, ptr %18, align 1, !tbaa !10
  %20 = and i8 %19, %17
  %21 = getelementptr inbounds nuw i8, ptr %6, i64 %11
  store i8 %20, ptr %21, align 1, !tbaa !10
  %22 = add nuw nsw i64 %11, 1
  %23 = tail call i32 @llvm.smax.i32(i32 %12, i32 8)
  %24 = add nsw i32 %23, -8
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %10

26:                                               ; preds = %10, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %struct.S, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(19) %1, i8 0, i64 19, i1 false)
  call void @foo(ptr noundef nonnull @__const.main.x, ptr noundef nonnull %1)
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 3
  %3 = load <4 x i8>, ptr %2, align 1
  %4 = freeze <4 x i8> %3
  %5 = bitcast <4 x i8> %4 to i32
  %6 = icmp eq i32 %5, -2134066262
  br i1 %6, label %8, label %7

7:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

8:                                                ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #5

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 1}
!7 = !{!"S", !8, i64 0, !8, i64 1, !8, i64 2, !8, i64 3}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
