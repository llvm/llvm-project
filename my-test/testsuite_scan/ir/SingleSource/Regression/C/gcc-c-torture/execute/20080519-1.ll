; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20080519-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20080519-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.du_chain = type { ptr, i32 }

@reg_class_contents = dso_local local_unnamed_addr global [2 x [2 x i64]] zeroinitializer, align 8

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @merge_overlapping_regs(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i64, ptr %0, align 8, !tbaa !6
  %3 = icmp eq i64 %2, -1
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !6
  %7 = icmp eq i64 %6, -1
  br i1 %7, label %9, label %8

8:                                                ; preds = %4, %1
  tail call void @abort() #5
  unreachable

9:                                                ; preds = %4
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @regrename_optimize(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = alloca [2 x i64], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  %3 = load ptr, ptr %0, align 8, !tbaa !10
  %4 = icmp eq ptr %3, null
  br i1 %4, label %26, label %5

5:                                                ; preds = %1, %5
  %6 = phi ptr [ %16, %5 ], [ %3, %1 ]
  %7 = phi ptr [ %6, %5 ], [ %0, %1 ]
  %8 = phi <2 x i64> [ %15, %5 ], [ zeroinitializer, %1 ]
  %9 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %10 = load i32, ptr %9, align 8, !tbaa !15
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds [2 x i64], ptr @reg_class_contents, i64 %11
  %13 = load <2 x i64>, ptr %12, align 8, !tbaa !6
  %14 = xor <2 x i64> %13, splat (i64 -1)
  %15 = or <2 x i64> %8, %14
  %16 = load ptr, ptr %6, align 8, !tbaa !10
  %17 = icmp eq ptr %16, null
  br i1 %17, label %18, label %5, !llvm.loop !16

18:                                               ; preds = %5
  %19 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %20 = load i32, ptr %19, align 8, !tbaa !15
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds [2 x i64], ptr @reg_class_contents, i64 %21
  %23 = load <2 x i64>, ptr %22, align 8, !tbaa !6
  %24 = xor <2 x i64> %23, splat (i64 -1)
  %25 = or <2 x i64> %15, %24
  store <2 x i64> %25, ptr %2, align 16, !tbaa !6
  call void @merge_overlapping_regs(ptr noundef nonnull %2)
  br label %26

26:                                               ; preds = %1, %18
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca %struct.du_chain, align 8
  %2 = alloca %struct.du_chain, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  store ptr %1, ptr %2, align 8, !tbaa !10
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store <2 x i32> <i32 1, i32 0>, ptr %3, align 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) @reg_class_contents, i8 -1, i64 16, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) getelementptr inbounds nuw (i8, ptr @reg_class_contents, i64 16), i8 0, i64 16, i1 false)
  call void @regrename_optimize(ptr noundef nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
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
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !12, i64 0}
!11 = !{!"du_chain", !12, i64 0, !14, i64 8}
!12 = !{!"p1 _ZTS8du_chain", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
!14 = !{!"int", !8, i64 0}
!15 = !{!11, !14, i64 8}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.mustprogress"}
