; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071030-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20071030-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.client_s = type { i32, i32, [64 x %struct.client_frame_t] }
%struct.client_frame_t = type { double, float, %struct.packet_entities_t }
%struct.packet_entities_t = type { i32, ptr }

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local i32 @CalcPing(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i32, ptr %0, align 8, !tbaa !6
  %3 = icmp eq i32 %2, 1
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %6 = load i32, ptr %5, align 4, !tbaa !11
  br label %31

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %9

9:                                                ; preds = %7, %9
  %10 = phi ptr [ %8, %7 ], [ %22, %9 ]
  %11 = phi i32 [ 0, %7 ], [ %21, %9 ]
  %12 = phi i32 [ 0, %7 ], [ %20, %9 ]
  %13 = phi float [ 0.000000e+00, %7 ], [ %18, %9 ]
  %14 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %15 = load float, ptr %14, align 8, !tbaa !12
  %16 = fcmp ogt float %15, 0.000000e+00
  %17 = fadd float %13, %15
  %18 = select i1 %16, float %17, float %13
  %19 = zext i1 %16 to i32
  %20 = add nuw nsw i32 %12, %19
  %21 = add nuw nsw i32 %11, 1
  %22 = getelementptr inbounds nuw i8, ptr %10, i64 32
  %23 = icmp eq i32 %21, 64
  br i1 %23, label %24, label %9, !llvm.loop !18

24:                                               ; preds = %9
  %25 = icmp eq i32 %20, 0
  br i1 %25, label %31, label %26

26:                                               ; preds = %24
  %27 = uitofp nneg i32 %20 to float
  %28 = fdiv float %18, %27
  %29 = fmul float %28, 1.000000e+03
  %30 = fptosi float %29 to i32
  br label %31

31:                                               ; preds = %24, %26, %4
  %32 = phi i32 [ %6, %4 ], [ %30, %26 ], [ 9999, %24 ]
  ret i32 %32
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %struct.client_s, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(2056) %1, i8 0, i64 2056, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store float 1.000000e+00, ptr %2, align 8, !tbaa !12
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 8
  br label %4

4:                                                ; preds = %4, %0
  %5 = phi ptr [ %3, %0 ], [ %17, %4 ]
  %6 = phi i32 [ 0, %0 ], [ %16, %4 ]
  %7 = phi i32 [ 0, %0 ], [ %15, %4 ]
  %8 = phi float [ 0.000000e+00, %0 ], [ %13, %4 ]
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %10 = load float, ptr %9, align 8, !tbaa !12
  %11 = fcmp ogt float %10, 0.000000e+00
  %12 = fadd float %8, %10
  %13 = select i1 %11, float %12, float %8
  %14 = zext i1 %11 to i32
  %15 = add nuw nsw i32 %7, %14
  %16 = add nuw nsw i32 %6, 1
  %17 = getelementptr inbounds nuw i8, ptr %5, i64 32
  %18 = icmp eq i32 %16, 64
  br i1 %18, label %19, label %4, !llvm.loop !18

19:                                               ; preds = %4
  %20 = icmp eq i32 %15, 0
  br i1 %20, label %27, label %21

21:                                               ; preds = %19
  %22 = uitofp nneg i32 %15 to float
  %23 = fdiv float %13, %22
  %24 = fmul float %23, 1.000000e+03
  %25 = fptosi float %24 to i32
  %26 = icmp eq i32 %25, 1000
  br i1 %26, label %28, label %27

27:                                               ; preds = %19, %21
  tail call void @abort() #6
  unreachable

28:                                               ; preds = %21
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !8, i64 0}
!7 = !{!"client_s", !8, i64 0, !8, i64 4, !9, i64 8}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 4}
!12 = !{!13, !15, i64 8}
!13 = !{!"", !14, i64 0, !15, i64 8, !16, i64 16}
!14 = !{!"double", !9, i64 0}
!15 = !{!"float", !9, i64 0}
!16 = !{!"", !8, i64 0, !17, i64 8}
!17 = !{!"any pointer", !9, i64 0}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
