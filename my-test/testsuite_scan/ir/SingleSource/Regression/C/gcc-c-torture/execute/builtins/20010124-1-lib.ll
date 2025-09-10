; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/20010124-1-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/20010124-1-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { [1024 x i8] }

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @f(ptr dead_on_unwind noalias writable writeonly sret(%struct.S) align 1 captures(none) initializes((0, 1024)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(1024) %0, ptr noundef nonnull align 1 dereferenceable(1024) %1, i64 1024, i1 false), !tbaa.struct !6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @g(ptr noundef readnone captures(none) %0) local_unnamed_addr #2 {
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef ptr @memcpy(ptr noundef returned writeonly captures(address, ret: address, provenance) %0, ptr noundef readonly captures(address) %1, i64 noundef %2) local_unnamed_addr #3 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = load i32, ptr @inside_main, align 4, !tbaa !10
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = icmp eq i64 %2, 0
  br i1 %9, label %77, label %10

10:                                               ; preds = %8
  %11 = icmp ult i64 %2, 8
  %12 = sub i64 %5, %4
  %13 = icmp ult i64 %12, 32
  %14 = or i1 %11, %13
  br i1 %14, label %15, label %19

15:                                               ; preds = %35, %54, %10
  %16 = phi ptr [ %1, %10 ], [ %36, %35 ], [ %44, %54 ]
  %17 = phi ptr [ %0, %10 ], [ %37, %35 ], [ %45, %54 ]
  %18 = phi i64 [ %2, %10 ], [ %38, %35 ], [ %46, %54 ]
  br label %68

19:                                               ; preds = %10
  %20 = icmp ult i64 %2, 32
  br i1 %20, label %41, label %21

21:                                               ; preds = %19
  %22 = and i64 %2, -32
  br label %23

23:                                               ; preds = %23, %21
  %24 = phi i64 [ 0, %21 ], [ %31, %23 ]
  %25 = getelementptr i8, ptr %1, i64 %24
  %26 = getelementptr i8, ptr %0, i64 %24
  %27 = getelementptr i8, ptr %25, i64 16
  %28 = load <16 x i8>, ptr %25, align 1, !tbaa !7
  %29 = load <16 x i8>, ptr %27, align 1, !tbaa !7
  %30 = getelementptr i8, ptr %26, i64 16
  store <16 x i8> %28, ptr %26, align 1, !tbaa !7
  store <16 x i8> %29, ptr %30, align 1, !tbaa !7
  %31 = add nuw i64 %24, 32
  %32 = icmp eq i64 %31, %22
  br i1 %32, label %33, label %23, !llvm.loop !12

33:                                               ; preds = %23
  %34 = icmp eq i64 %2, %22
  br i1 %34, label %77, label %35

35:                                               ; preds = %33
  %36 = getelementptr i8, ptr %1, i64 %22
  %37 = getelementptr i8, ptr %0, i64 %22
  %38 = and i64 %2, 31
  %39 = and i64 %2, 24
  %40 = icmp eq i64 %39, 0
  br i1 %40, label %15, label %41

41:                                               ; preds = %35, %19
  %42 = phi i64 [ %22, %35 ], [ 0, %19 ]
  %43 = and i64 %2, -8
  %44 = getelementptr i8, ptr %1, i64 %43
  %45 = getelementptr i8, ptr %0, i64 %43
  %46 = and i64 %2, 7
  br label %47

47:                                               ; preds = %47, %41
  %48 = phi i64 [ %42, %41 ], [ %52, %47 ]
  %49 = getelementptr i8, ptr %1, i64 %48
  %50 = getelementptr i8, ptr %0, i64 %48
  %51 = load <8 x i8>, ptr %49, align 1, !tbaa !7
  store <8 x i8> %51, ptr %50, align 1, !tbaa !7
  %52 = add nuw i64 %48, 8
  %53 = icmp eq i64 %52, %43
  br i1 %53, label %54, label %47, !llvm.loop !16

54:                                               ; preds = %47
  %55 = icmp eq i64 %2, %43
  br i1 %55, label %77, label %15

56:                                               ; preds = %3
  %57 = icmp ult ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 %2
  %59 = icmp ugt ptr %58, %1
  %60 = select i1 %57, i1 %59, i1 false
  br i1 %60, label %61, label %62

61:                                               ; preds = %56
  tail call void @abort() #5
  unreachable

62:                                               ; preds = %56
  %63 = icmp ult ptr %1, %0
  %64 = getelementptr inbounds nuw i8, ptr %1, i64 %2
  %65 = icmp ugt ptr %64, %0
  %66 = select i1 %63, i1 %65, i1 false
  br i1 %66, label %67, label %77

67:                                               ; preds = %62
  tail call void @abort() #5
  unreachable

68:                                               ; preds = %15, %68
  %69 = phi ptr [ %73, %68 ], [ %16, %15 ]
  %70 = phi ptr [ %75, %68 ], [ %17, %15 ]
  %71 = phi i64 [ %72, %68 ], [ %18, %15 ]
  %72 = add i64 %71, -1
  %73 = getelementptr inbounds nuw i8, ptr %69, i64 1
  %74 = load i8, ptr %69, align 1, !tbaa !7
  %75 = getelementptr inbounds nuw i8, ptr %70, i64 1
  store i8 %74, ptr %70, align 1, !tbaa !7
  %76 = icmp eq i64 %72, 0
  br i1 %76, label %77, label %68, !llvm.loop !17

77:                                               ; preds = %68, %33, %54, %8, %62
  ret ptr %0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 0, i64 1024, !7}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13, !14, !15}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.isvectorized", i32 1}
!15 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = distinct !{!16, !13, !14, !15}
!17 = distinct !{!17, !13, !14}
