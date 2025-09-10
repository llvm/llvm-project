; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memmove-2-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memmove-2-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef ptr @memmove(ptr noundef returned writeonly captures(address, ret: address, provenance) %0, ptr noundef readonly captures(address) %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = ptrtoint ptr %0 to i64
  %5 = ptrtoint ptr %1 to i64
  %6 = ptrtoint ptr %0 to i64
  %7 = load i32, ptr @inside_main, align 4, !tbaa !6
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @abort() #4
  unreachable

10:                                               ; preds = %3
  %11 = icmp ult ptr %1, %0
  %12 = icmp eq i64 %2, 0
  br i1 %11, label %60, label %13

13:                                               ; preds = %10
  br i1 %12, label %125, label %14

14:                                               ; preds = %13
  %15 = icmp ult i64 %2, 8
  %16 = sub i64 %6, %5
  %17 = icmp ult i64 %16, 32
  %18 = or i1 %15, %17
  br i1 %18, label %19, label %23

19:                                               ; preds = %39, %58, %14
  %20 = phi ptr [ %1, %14 ], [ %40, %39 ], [ %48, %58 ]
  %21 = phi ptr [ %0, %14 ], [ %41, %39 ], [ %49, %58 ]
  %22 = phi i64 [ %2, %14 ], [ %42, %39 ], [ %50, %58 ]
  br label %116

23:                                               ; preds = %14
  %24 = icmp ult i64 %2, 32
  br i1 %24, label %45, label %25

25:                                               ; preds = %23
  %26 = and i64 %2, -32
  br label %27

27:                                               ; preds = %27, %25
  %28 = phi i64 [ 0, %25 ], [ %35, %27 ]
  %29 = getelementptr i8, ptr %1, i64 %28
  %30 = getelementptr i8, ptr %0, i64 %28
  %31 = getelementptr i8, ptr %29, i64 16
  %32 = load <16 x i8>, ptr %29, align 1, !tbaa !10
  %33 = load <16 x i8>, ptr %31, align 1, !tbaa !10
  %34 = getelementptr i8, ptr %30, i64 16
  store <16 x i8> %32, ptr %30, align 1, !tbaa !10
  store <16 x i8> %33, ptr %34, align 1, !tbaa !10
  %35 = add nuw i64 %28, 32
  %36 = icmp eq i64 %35, %26
  br i1 %36, label %37, label %27, !llvm.loop !11

37:                                               ; preds = %27
  %38 = icmp eq i64 %2, %26
  br i1 %38, label %125, label %39

39:                                               ; preds = %37
  %40 = getelementptr i8, ptr %1, i64 %26
  %41 = getelementptr i8, ptr %0, i64 %26
  %42 = and i64 %2, 31
  %43 = and i64 %2, 24
  %44 = icmp eq i64 %43, 0
  br i1 %44, label %19, label %45

45:                                               ; preds = %39, %23
  %46 = phi i64 [ %26, %39 ], [ 0, %23 ]
  %47 = and i64 %2, -8
  %48 = getelementptr i8, ptr %1, i64 %47
  %49 = getelementptr i8, ptr %0, i64 %47
  %50 = and i64 %2, 7
  br label %51

51:                                               ; preds = %51, %45
  %52 = phi i64 [ %46, %45 ], [ %56, %51 ]
  %53 = getelementptr i8, ptr %1, i64 %52
  %54 = getelementptr i8, ptr %0, i64 %52
  %55 = load <8 x i8>, ptr %53, align 1, !tbaa !10
  store <8 x i8> %55, ptr %54, align 1, !tbaa !10
  %56 = add nuw i64 %52, 8
  %57 = icmp eq i64 %56, %47
  br i1 %57, label %58, label %51, !llvm.loop !15

58:                                               ; preds = %51
  %59 = icmp eq i64 %2, %47
  br i1 %59, label %125, label %19

60:                                               ; preds = %10
  br i1 %12, label %125, label %61

61:                                               ; preds = %60
  %62 = icmp ult i64 %2, 8
  %63 = sub i64 %5, %4
  %64 = icmp ult i64 %63, 32
  %65 = or i1 %62, %64
  br i1 %65, label %107, label %66

66:                                               ; preds = %61
  %67 = icmp ult i64 %2, 32
  br i1 %67, label %90, label %68

68:                                               ; preds = %66
  %69 = and i64 %2, -32
  br label %70

70:                                               ; preds = %70, %68
  %71 = phi i64 [ 0, %68 ], [ %82, %70 ]
  %72 = xor i64 %71, -1
  %73 = add i64 %2, %72
  %74 = getelementptr inbounds nuw i8, ptr %1, i64 %73
  %75 = getelementptr inbounds i8, ptr %74, i64 -15
  %76 = getelementptr inbounds i8, ptr %74, i64 -31
  %77 = load <16 x i8>, ptr %75, align 1, !tbaa !10
  %78 = load <16 x i8>, ptr %76, align 1, !tbaa !10
  %79 = getelementptr inbounds nuw i8, ptr %0, i64 %73
  %80 = getelementptr inbounds i8, ptr %79, i64 -15
  %81 = getelementptr inbounds i8, ptr %79, i64 -31
  store <16 x i8> %77, ptr %80, align 1, !tbaa !10
  store <16 x i8> %78, ptr %81, align 1, !tbaa !10
  %82 = add nuw i64 %71, 32
  %83 = icmp eq i64 %82, %69
  br i1 %83, label %84, label %70, !llvm.loop !16

84:                                               ; preds = %70
  %85 = icmp eq i64 %2, %69
  br i1 %85, label %125, label %86

86:                                               ; preds = %84
  %87 = and i64 %2, 31
  %88 = and i64 %2, 24
  %89 = icmp eq i64 %88, 0
  br i1 %89, label %107, label %90

90:                                               ; preds = %86, %66
  %91 = phi i64 [ %69, %86 ], [ 0, %66 ]
  %92 = and i64 %2, -8
  %93 = and i64 %2, 7
  br label %94

94:                                               ; preds = %94, %90
  %95 = phi i64 [ %91, %90 ], [ %103, %94 ]
  %96 = xor i64 %95, -1
  %97 = add i64 %2, %96
  %98 = getelementptr inbounds nuw i8, ptr %1, i64 %97
  %99 = getelementptr inbounds i8, ptr %98, i64 -7
  %100 = load <8 x i8>, ptr %99, align 1, !tbaa !10
  %101 = getelementptr inbounds nuw i8, ptr %0, i64 %97
  %102 = getelementptr inbounds i8, ptr %101, i64 -7
  store <8 x i8> %100, ptr %102, align 1, !tbaa !10
  %103 = add nuw i64 %95, 8
  %104 = icmp eq i64 %103, %92
  br i1 %104, label %105, label %94, !llvm.loop !17

105:                                              ; preds = %94
  %106 = icmp eq i64 %2, %92
  br i1 %106, label %125, label %107

107:                                              ; preds = %86, %105, %61
  %108 = phi i64 [ %2, %61 ], [ %87, %86 ], [ %93, %105 ]
  br label %109

109:                                              ; preds = %107, %109
  %110 = phi i64 [ %111, %109 ], [ %108, %107 ]
  %111 = add i64 %110, -1
  %112 = getelementptr inbounds nuw i8, ptr %1, i64 %111
  %113 = load i8, ptr %112, align 1, !tbaa !10
  %114 = getelementptr inbounds nuw i8, ptr %0, i64 %111
  store i8 %113, ptr %114, align 1, !tbaa !10
  %115 = icmp eq i64 %111, 0
  br i1 %115, label %125, label %109, !llvm.loop !18

116:                                              ; preds = %19, %116
  %117 = phi ptr [ %121, %116 ], [ %20, %19 ]
  %118 = phi ptr [ %123, %116 ], [ %21, %19 ]
  %119 = phi i64 [ %120, %116 ], [ %22, %19 ]
  %120 = add i64 %119, -1
  %121 = getelementptr inbounds nuw i8, ptr %117, i64 1
  %122 = load i8, ptr %117, align 1, !tbaa !10
  %123 = getelementptr inbounds nuw i8, ptr %118, i64 1
  store i8 %122, ptr %118, align 1, !tbaa !10
  %124 = icmp eq i64 %120, 0
  br i1 %124, label %125, label %116, !llvm.loop !19

125:                                              ; preds = %116, %109, %37, %58, %84, %105, %13, %60
  ret ptr %0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @bcopy(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) %1, i64 noundef %2) local_unnamed_addr #2 {
  tail call void @llvm.memmove.p0.p0.i64(ptr align 1 %1, ptr align 1 %0, i64 %2, i1 false)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #3

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { noreturn nounwind }

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
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12, !13, !14}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
!15 = distinct !{!15, !12, !13, !14}
!16 = distinct !{!16, !12, !13, !14}
!17 = distinct !{!17, !12, !13, !14}
!18 = distinct !{!18, !12, !13}
!19 = distinct !{!19, !12, !13}
