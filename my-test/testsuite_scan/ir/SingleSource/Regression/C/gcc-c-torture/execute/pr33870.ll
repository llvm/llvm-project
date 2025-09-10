; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33870.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33870.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.PgHdr = type { i32, ptr, ptr, ptr, ptr, ptr, i8, i16, ptr, ptr, i32 }

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local ptr @sort_pagelist(ptr noundef %0) local_unnamed_addr #0 {
  %2 = alloca %struct.PgHdr, align 8
  %3 = alloca %struct.PgHdr, align 8
  %4 = alloca %struct.PgHdr, align 8
  %5 = alloca [25 x ptr], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(200) %5, i8 0, i64 200, i1 false)
  %6 = icmp eq ptr %0, null
  br i1 %6, label %93, label %7

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 56
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 192
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 56
  br label %11

11:                                               ; preds = %7, %89
  %12 = phi ptr [ %0, %7 ], [ %14, %89 ]
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 56
  %14 = load ptr, ptr %13, align 8, !tbaa !6
  store ptr null, ptr %13, align 8, !tbaa !6
  br label %15

15:                                               ; preds = %11, %48
  %16 = phi i64 [ 0, %11 ], [ %53, %48 ]
  %17 = phi ptr [ %12, %11 ], [ %52, %48 ]
  %18 = getelementptr inbounds nuw ptr, ptr %5, i64 %16
  %19 = load ptr, ptr %18, align 8, !tbaa !14
  %20 = icmp eq ptr %19, null
  br i1 %20, label %21, label %22

21:                                               ; preds = %15
  store ptr %17, ptr %18, align 8, !tbaa !14
  br label %89

22:                                               ; preds = %15
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  %23 = icmp eq ptr %17, null
  br i1 %23, label %48, label %24

24:                                               ; preds = %22, %39
  %25 = phi ptr [ %42, %39 ], [ %4, %22 ]
  %26 = phi ptr [ %41, %39 ], [ %17, %22 ]
  %27 = phi ptr [ %40, %39 ], [ %19, %22 ]
  %28 = load i32, ptr %27, align 8, !tbaa !15
  %29 = load i32, ptr %26, align 8, !tbaa !15
  %30 = icmp ult i32 %28, %29
  %31 = getelementptr inbounds nuw i8, ptr %25, i64 56
  br i1 %30, label %32, label %36

32:                                               ; preds = %24
  store ptr %27, ptr %31, align 8, !tbaa !6
  %33 = getelementptr inbounds nuw i8, ptr %27, i64 56
  %34 = load ptr, ptr %33, align 8, !tbaa !6
  %35 = freeze ptr %34
  br label %39

36:                                               ; preds = %24
  store ptr %26, ptr %31, align 8, !tbaa !6
  %37 = getelementptr inbounds nuw i8, ptr %26, i64 56
  %38 = load ptr, ptr %37, align 8, !tbaa !6
  br label %39

39:                                               ; preds = %36, %32
  %40 = phi ptr [ %35, %32 ], [ %27, %36 ]
  %41 = phi ptr [ %26, %32 ], [ %38, %36 ]
  %42 = phi ptr [ %27, %32 ], [ %26, %36 ]
  %43 = icmp ne ptr %40, null
  %44 = icmp ne ptr %41, null
  %45 = select i1 %43, i1 %44, i1 false
  br i1 %45, label %24, label %46, !llvm.loop !16

46:                                               ; preds = %39
  %47 = select i1 %43, ptr %40, ptr %41
  br label %48

48:                                               ; preds = %22, %46
  %49 = phi ptr [ %42, %46 ], [ %4, %22 ]
  %50 = phi ptr [ %47, %46 ], [ %19, %22 ]
  %51 = getelementptr inbounds nuw i8, ptr %49, i64 56
  store ptr %50, ptr %51, align 8, !tbaa !6
  %52 = load ptr, ptr %8, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #5
  store ptr null, ptr %18, align 8, !tbaa !14
  %53 = add nuw nsw i64 %16, 1
  %54 = icmp eq i64 %53, 24
  br i1 %54, label %55, label %15, !llvm.loop !18

55:                                               ; preds = %48
  %56 = load ptr, ptr %9, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  %57 = icmp ne ptr %56, null
  %58 = icmp ne ptr %52, null
  %59 = and i1 %58, %57
  br i1 %59, label %60, label %81

60:                                               ; preds = %55, %74
  %61 = phi ptr [ %77, %74 ], [ %3, %55 ]
  %62 = phi ptr [ %76, %74 ], [ %52, %55 ]
  %63 = phi ptr [ %75, %74 ], [ %56, %55 ]
  %64 = load i32, ptr %63, align 8, !tbaa !15
  %65 = load i32, ptr %62, align 8, !tbaa !15
  %66 = icmp ult i32 %64, %65
  %67 = getelementptr inbounds nuw i8, ptr %61, i64 56
  br i1 %66, label %68, label %71

68:                                               ; preds = %60
  store ptr %63, ptr %67, align 8, !tbaa !6
  %69 = getelementptr inbounds nuw i8, ptr %63, i64 56
  %70 = load ptr, ptr %69, align 8, !tbaa !6
  br label %74

71:                                               ; preds = %60
  store ptr %62, ptr %67, align 8, !tbaa !6
  %72 = getelementptr inbounds nuw i8, ptr %62, i64 56
  %73 = load ptr, ptr %72, align 8, !tbaa !6
  br label %74

74:                                               ; preds = %71, %68
  %75 = phi ptr [ %70, %68 ], [ %63, %71 ]
  %76 = phi ptr [ %62, %68 ], [ %73, %71 ]
  %77 = phi ptr [ %63, %68 ], [ %62, %71 ]
  %78 = icmp ne ptr %75, null
  %79 = icmp ne ptr %76, null
  %80 = select i1 %78, i1 %79, i1 false
  br i1 %80, label %60, label %81, !llvm.loop !16

81:                                               ; preds = %74, %55
  %82 = phi ptr [ %56, %55 ], [ %75, %74 ]
  %83 = phi ptr [ %3, %55 ], [ %77, %74 ]
  %84 = phi i1 [ %57, %55 ], [ %78, %74 ]
  %85 = phi ptr [ %52, %55 ], [ %76, %74 ]
  %86 = getelementptr inbounds nuw i8, ptr %83, i64 56
  %87 = select i1 %84, ptr %82, ptr %85
  store ptr %87, ptr %86, align 8, !tbaa !6
  %88 = load ptr, ptr %10, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  store ptr %88, ptr %9, align 8, !tbaa !14
  br label %89

89:                                               ; preds = %21, %81
  %90 = icmp eq ptr %14, null
  br i1 %90, label %91, label %11, !llvm.loop !19

91:                                               ; preds = %89
  %92 = load ptr, ptr %5, align 8, !tbaa !14
  br label %93

93:                                               ; preds = %91, %1
  %94 = phi ptr [ %92, %91 ], [ null, %1 ]
  %95 = getelementptr inbounds nuw i8, ptr %2, i64 56
  br label %96

96:                                               ; preds = %93, %125
  %97 = phi i64 [ 1, %93 ], [ %133, %125 ]
  %98 = phi ptr [ %94, %93 ], [ %132, %125 ]
  %99 = getelementptr inbounds nuw ptr, ptr %5, i64 %97
  %100 = load ptr, ptr %99, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  %101 = icmp ne ptr %98, null
  %102 = icmp ne ptr %100, null
  %103 = and i1 %101, %102
  br i1 %103, label %104, label %125

104:                                              ; preds = %96, %118
  %105 = phi ptr [ %121, %118 ], [ %2, %96 ]
  %106 = phi ptr [ %120, %118 ], [ %100, %96 ]
  %107 = phi ptr [ %119, %118 ], [ %98, %96 ]
  %108 = load i32, ptr %107, align 8, !tbaa !15
  %109 = load i32, ptr %106, align 8, !tbaa !15
  %110 = icmp ult i32 %108, %109
  %111 = getelementptr inbounds nuw i8, ptr %105, i64 56
  br i1 %110, label %112, label %115

112:                                              ; preds = %104
  store ptr %107, ptr %111, align 8, !tbaa !6
  %113 = getelementptr inbounds nuw i8, ptr %107, i64 56
  %114 = load ptr, ptr %113, align 8, !tbaa !6
  br label %118

115:                                              ; preds = %104
  store ptr %106, ptr %111, align 8, !tbaa !6
  %116 = getelementptr inbounds nuw i8, ptr %106, i64 56
  %117 = load ptr, ptr %116, align 8, !tbaa !6
  br label %118

118:                                              ; preds = %115, %112
  %119 = phi ptr [ %114, %112 ], [ %107, %115 ]
  %120 = phi ptr [ %106, %112 ], [ %117, %115 ]
  %121 = phi ptr [ %107, %112 ], [ %106, %115 ]
  %122 = icmp ne ptr %119, null
  %123 = icmp ne ptr %120, null
  %124 = select i1 %122, i1 %123, i1 false
  br i1 %124, label %104, label %125, !llvm.loop !16

125:                                              ; preds = %118, %96
  %126 = phi ptr [ %98, %96 ], [ %119, %118 ]
  %127 = phi ptr [ %2, %96 ], [ %121, %118 ]
  %128 = phi i1 [ %101, %96 ], [ %122, %118 ]
  %129 = phi ptr [ %100, %96 ], [ %120, %118 ]
  %130 = getelementptr inbounds nuw i8, ptr %127, i64 56
  %131 = select i1 %128, ptr %126, ptr %129
  store ptr %131, ptr %130, align 8, !tbaa !6
  %132 = load ptr, ptr %95, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  %133 = add nuw nsw i64 %97, 1
  %134 = icmp eq i64 %133, 25
  br i1 %134, label %135, label %96, !llvm.loop !20

135:                                              ; preds = %125
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  ret ptr %132
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca [5 x %struct.PgHdr], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store i32 5, ptr %1, align 8, !tbaa !15
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 56
  store ptr %2, ptr %3, align 8, !tbaa !6
  store i32 4, ptr %2, align 8, !tbaa !15
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 136
  store ptr %4, ptr %5, align 8, !tbaa !6
  store i32 1, ptr %4, align 8, !tbaa !15
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 216
  store ptr %6, ptr %7, align 8, !tbaa !6
  store i32 3, ptr %6, align 8, !tbaa !15
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 296
  store ptr null, ptr %8, align 8, !tbaa !6
  %9 = call ptr @sort_pagelist(ptr noundef nonnull %1)
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 56
  %11 = load ptr, ptr %10, align 8, !tbaa !6
  %12 = icmp eq ptr %11, %9
  br i1 %12, label %13, label %14

13:                                               ; preds = %0
  call void @abort() #6
  unreachable

14:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !11, i64 56}
!7 = !{!"PgHdr", !8, i64 0, !11, i64 8, !11, i64 16, !11, i64 24, !11, i64 32, !11, i64 40, !9, i64 48, !13, i64 50, !11, i64 56, !11, i64 64, !8, i64 72}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"p1 _ZTS5PgHdr", !12, i64 0}
!12 = !{!"any pointer", !9, i64 0}
!13 = !{!"short", !9, i64 0}
!14 = !{!11, !11, i64 0}
!15 = !{!7, !8, i64 0}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.mustprogress"}
!18 = distinct !{!18, !17}
!19 = distinct !{!19, !17}
!20 = distinct !{!20, !17}
