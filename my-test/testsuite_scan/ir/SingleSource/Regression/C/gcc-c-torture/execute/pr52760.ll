; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr52760.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr52760.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.T = type { i16, i16, i16, i16 }

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @foo(i32 noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %27

4:                                                ; preds = %2
  %5 = zext nneg i32 %0 to i64
  %6 = icmp ult i32 %0, 8
  br i1 %6, label %18, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 2147483640
  br label %9

9:                                                ; preds = %9, %7
  %10 = phi i64 [ 0, %7 ], [ %14, %9 ]
  %11 = getelementptr inbounds nuw %struct.T, ptr %1, i64 %10
  %12 = load <32 x i16>, ptr %11, align 2, !tbaa !6
  %13 = tail call <32 x i16> @llvm.bswap.v32i16(<32 x i16> %12)
  store <32 x i16> %13, ptr %11, align 2, !tbaa !6
  %14 = add nuw i64 %10, 8
  %15 = icmp eq i64 %14, %8
  br i1 %15, label %16, label %9, !llvm.loop !10

16:                                               ; preds = %9
  %17 = icmp eq i64 %8, %5
  br i1 %17, label %27, label %18

18:                                               ; preds = %4, %16
  %19 = phi i64 [ 0, %4 ], [ %8, %16 ]
  br label %20

20:                                               ; preds = %18, %20
  %21 = phi i64 [ %25, %20 ], [ %19, %18 ]
  %22 = getelementptr inbounds nuw %struct.T, ptr %1, i64 %21
  %23 = load <4 x i16>, ptr %22, align 2, !tbaa !6
  %24 = tail call <4 x i16> @llvm.bswap.v4i16(<4 x i16> %23)
  store <4 x i16> %24, ptr %22, align 2, !tbaa !6
  %25 = add nuw nsw i64 %21, 1
  %26 = icmp eq i64 %25, %5
  br i1 %26, label %27, label %20, !llvm.loop !14

27:                                               ; preds = %20, %16, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %struct.T, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store i64 434320308619640833, ptr %1, align 8
  call void @foo(i32 noundef 1, ptr noundef nonnull %1)
  %2 = load <4 x i16>, ptr %1, align 8
  %3 = freeze <4 x i16> %2
  %4 = bitcast <4 x i16> %3 to i64
  %5 = icmp eq i64 %4, 506097522914230528
  br i1 %5, label %7, label %6

6:                                                ; preds = %0
  tail call void @abort() #6
  unreachable

7:                                                ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <32 x i16> @llvm.bswap.v32i16(<32 x i16>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x i16> @llvm.bswap.v4i16(<4 x i16>) #4

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !13, !12}
