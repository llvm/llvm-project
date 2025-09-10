; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33870-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr33870-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.PgHdr = type { i32, %struct.anon }
%struct.anon = type { i32, ptr, ptr, ptr, ptr, ptr, i8, i16, ptr, ptr, i32 }

@xx = dso_local local_unnamed_addr global ptr null, align 8
@vx = dso_local global i32 0, align 4

; Function Attrs: nofree noinline norecurse nounwind uwtable
define dso_local ptr @sort_pagelist(ptr noundef %0) local_unnamed_addr #0 {
  %2 = alloca %struct.PgHdr, align 8
  %3 = alloca %struct.PgHdr, align 8
  %4 = alloca %struct.PgHdr, align 8
  %5 = alloca [25 x ptr], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(200) %5, i8 0, i64 200, i1 false)
  %6 = icmp eq ptr %0, null
  br i1 %6, label %101, label %7

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 64
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 192
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 64
  br label %11

11:                                               ; preds = %7, %97
  %12 = phi ptr [ %0, %7 ], [ %14, %97 ]
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 64
  %14 = load ptr, ptr %13, align 8, !tbaa !6
  store ptr null, ptr %13, align 8, !tbaa !6
  br label %15

15:                                               ; preds = %11, %52
  %16 = phi i64 [ 0, %11 ], [ %57, %52 ]
  %17 = phi ptr [ %12, %11 ], [ %56, %52 ]
  %18 = getelementptr inbounds nuw ptr, ptr %5, i64 %16
  %19 = load ptr, ptr %18, align 8, !tbaa !15
  %20 = icmp eq ptr %19, null
  br i1 %20, label %21, label %22

21:                                               ; preds = %15
  store ptr %17, ptr %18, align 8, !tbaa !15
  br label %97

22:                                               ; preds = %15
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  %23 = icmp eq ptr %17, null
  br i1 %23, label %52, label %24

24:                                               ; preds = %22, %40
  %25 = phi ptr [ %43, %40 ], [ %4, %22 ]
  %26 = phi ptr [ %42, %40 ], [ %17, %22 ]
  %27 = phi ptr [ %44, %40 ], [ %19, %22 ]
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 8
  %29 = load i32, ptr %28, align 8, !tbaa !16
  %30 = getelementptr inbounds nuw i8, ptr %26, i64 8
  %31 = load i32, ptr %30, align 8, !tbaa !16
  %32 = icmp ult i32 %29, %31
  %33 = getelementptr inbounds nuw i8, ptr %25, i64 64
  br i1 %32, label %34, label %37

34:                                               ; preds = %24
  store ptr %27, ptr %33, align 8, !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %27, i64 64
  %36 = load ptr, ptr %35, align 8, !tbaa !6
  br label %40

37:                                               ; preds = %24
  store ptr %26, ptr %33, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %26, i64 64
  %39 = load ptr, ptr %38, align 8, !tbaa !6
  br label %40

40:                                               ; preds = %37, %34
  %41 = phi ptr [ %36, %34 ], [ %27, %37 ]
  %42 = phi ptr [ %26, %34 ], [ %39, %37 ]
  %43 = phi ptr [ %27, %34 ], [ %26, %37 ]
  %44 = freeze ptr %41
  %45 = load ptr, ptr %8, align 8, !tbaa !15
  %46 = load i32, ptr %45, align 8, !tbaa !17
  store volatile i32 %46, ptr @vx, align 4, !tbaa !18
  %47 = icmp ne ptr %44, null
  %48 = icmp ne ptr %42, null
  %49 = select i1 %47, i1 %48, i1 false
  br i1 %49, label %24, label %50, !llvm.loop !19

50:                                               ; preds = %40
  %51 = select i1 %47, ptr %44, ptr %42
  br label %52

52:                                               ; preds = %22, %50
  %53 = phi ptr [ %43, %50 ], [ %4, %22 ]
  %54 = phi ptr [ %51, %50 ], [ %19, %22 ]
  %55 = getelementptr inbounds nuw i8, ptr %53, i64 64
  store ptr %54, ptr %55, align 8, !tbaa !6
  %56 = load ptr, ptr %8, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #5
  store ptr null, ptr %18, align 8, !tbaa !15
  %57 = add nuw nsw i64 %16, 1
  %58 = icmp eq i64 %57, 24
  br i1 %58, label %59, label %15, !llvm.loop !21

59:                                               ; preds = %52
  %60 = load ptr, ptr %9, align 8, !tbaa !15
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  store ptr %10, ptr @xx, align 8, !tbaa !22
  %61 = icmp ne ptr %60, null
  %62 = icmp ne ptr %56, null
  %63 = and i1 %62, %61
  br i1 %63, label %64, label %89

64:                                               ; preds = %59, %80
  %65 = phi ptr [ %83, %80 ], [ %3, %59 ]
  %66 = phi ptr [ %82, %80 ], [ %56, %59 ]
  %67 = phi ptr [ %81, %80 ], [ %60, %59 ]
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = load i32, ptr %68, align 8, !tbaa !16
  %70 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %71 = load i32, ptr %70, align 8, !tbaa !16
  %72 = icmp ult i32 %69, %71
  %73 = getelementptr inbounds nuw i8, ptr %65, i64 64
  br i1 %72, label %74, label %77

74:                                               ; preds = %64
  store ptr %67, ptr %73, align 8, !tbaa !6
  %75 = getelementptr inbounds nuw i8, ptr %67, i64 64
  %76 = load ptr, ptr %75, align 8, !tbaa !6
  br label %80

77:                                               ; preds = %64
  store ptr %66, ptr %73, align 8, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %66, i64 64
  %79 = load ptr, ptr %78, align 8, !tbaa !6
  br label %80

80:                                               ; preds = %77, %74
  %81 = phi ptr [ %76, %74 ], [ %67, %77 ]
  %82 = phi ptr [ %66, %74 ], [ %79, %77 ]
  %83 = phi ptr [ %67, %74 ], [ %66, %77 ]
  %84 = load ptr, ptr %10, align 8, !tbaa !15
  %85 = load i32, ptr %84, align 8, !tbaa !17
  store volatile i32 %85, ptr @vx, align 4, !tbaa !18
  %86 = icmp ne ptr %81, null
  %87 = icmp ne ptr %82, null
  %88 = select i1 %86, i1 %87, i1 false
  br i1 %88, label %64, label %89, !llvm.loop !19

89:                                               ; preds = %80, %59
  %90 = phi ptr [ %60, %59 ], [ %81, %80 ]
  %91 = phi ptr [ %3, %59 ], [ %83, %80 ]
  %92 = phi i1 [ %61, %59 ], [ %86, %80 ]
  %93 = phi ptr [ %56, %59 ], [ %82, %80 ]
  %94 = getelementptr inbounds nuw i8, ptr %91, i64 64
  %95 = select i1 %92, ptr %90, ptr %93
  store ptr %95, ptr %94, align 8, !tbaa !6
  %96 = load ptr, ptr %10, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  store ptr %96, ptr %9, align 8, !tbaa !15
  br label %97

97:                                               ; preds = %21, %89
  %98 = icmp eq ptr %14, null
  br i1 %98, label %99, label %11, !llvm.loop !25

99:                                               ; preds = %97
  %100 = load ptr, ptr %5, align 8, !tbaa !15
  br label %101

101:                                              ; preds = %99, %1
  %102 = phi ptr [ %100, %99 ], [ null, %1 ]
  %103 = getelementptr inbounds nuw i8, ptr %2, i64 64
  br label %104

104:                                              ; preds = %101, %137
  %105 = phi i64 [ 1, %101 ], [ %145, %137 ]
  %106 = phi ptr [ %102, %101 ], [ %144, %137 ]
  %107 = getelementptr inbounds nuw ptr, ptr %5, i64 %105
  %108 = load ptr, ptr %107, align 8, !tbaa !15
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  store ptr %103, ptr @xx, align 8, !tbaa !22
  %109 = icmp ne ptr %106, null
  %110 = icmp ne ptr %108, null
  %111 = and i1 %109, %110
  br i1 %111, label %112, label %137

112:                                              ; preds = %104, %128
  %113 = phi ptr [ %131, %128 ], [ %2, %104 ]
  %114 = phi ptr [ %130, %128 ], [ %108, %104 ]
  %115 = phi ptr [ %129, %128 ], [ %106, %104 ]
  %116 = getelementptr inbounds nuw i8, ptr %115, i64 8
  %117 = load i32, ptr %116, align 8, !tbaa !16
  %118 = getelementptr inbounds nuw i8, ptr %114, i64 8
  %119 = load i32, ptr %118, align 8, !tbaa !16
  %120 = icmp ult i32 %117, %119
  %121 = getelementptr inbounds nuw i8, ptr %113, i64 64
  br i1 %120, label %122, label %125

122:                                              ; preds = %112
  store ptr %115, ptr %121, align 8, !tbaa !6
  %123 = getelementptr inbounds nuw i8, ptr %115, i64 64
  %124 = load ptr, ptr %123, align 8, !tbaa !6
  br label %128

125:                                              ; preds = %112
  store ptr %114, ptr %121, align 8, !tbaa !6
  %126 = getelementptr inbounds nuw i8, ptr %114, i64 64
  %127 = load ptr, ptr %126, align 8, !tbaa !6
  br label %128

128:                                              ; preds = %125, %122
  %129 = phi ptr [ %124, %122 ], [ %115, %125 ]
  %130 = phi ptr [ %114, %122 ], [ %127, %125 ]
  %131 = phi ptr [ %115, %122 ], [ %114, %125 ]
  %132 = load ptr, ptr %103, align 8, !tbaa !15
  %133 = load i32, ptr %132, align 8, !tbaa !17
  store volatile i32 %133, ptr @vx, align 4, !tbaa !18
  %134 = icmp ne ptr %129, null
  %135 = icmp ne ptr %130, null
  %136 = select i1 %134, i1 %135, i1 false
  br i1 %136, label %112, label %137, !llvm.loop !19

137:                                              ; preds = %128, %104
  %138 = phi ptr [ %106, %104 ], [ %129, %128 ]
  %139 = phi ptr [ %2, %104 ], [ %131, %128 ]
  %140 = phi i1 [ %109, %104 ], [ %134, %128 ]
  %141 = phi ptr [ %108, %104 ], [ %130, %128 ]
  %142 = getelementptr inbounds nuw i8, ptr %139, i64 64
  %143 = select i1 %140, ptr %138, ptr %141
  store ptr %143, ptr %142, align 8, !tbaa !6
  %144 = load ptr, ptr %103, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  %145 = add nuw nsw i64 %105, 1
  %146 = icmp eq i64 %145, 25
  br i1 %146, label %147, label %104, !llvm.loop !26

147:                                              ; preds = %137
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  ret ptr %144
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
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 5, ptr %2, align 8, !tbaa !16
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 64
  store ptr %3, ptr %4, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 96
  store i32 4, ptr %5, align 8, !tbaa !16
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 152
  store ptr %6, ptr %7, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 184
  store i32 1, ptr %8, align 8, !tbaa !16
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 264
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 240
  store ptr %9, ptr %10, align 8, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 272
  store i32 3, ptr %11, align 8, !tbaa !16
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 328
  store ptr null, ptr %12, align 8, !tbaa !6
  %13 = call ptr @sort_pagelist(ptr noundef nonnull %1)
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 64
  %15 = load ptr, ptr %14, align 8, !tbaa !6
  %16 = icmp eq ptr %15, %13
  br i1 %16, label %17, label %18

17:                                               ; preds = %0
  call void @abort() #6
  unreachable

18:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { nofree noinline norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !12, i64 64}
!7 = !{!"PgHdr", !8, i64 0, !11, i64 8}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"", !8, i64 0, !12, i64 8, !12, i64 16, !12, i64 24, !12, i64 32, !12, i64 40, !9, i64 48, !14, i64 50, !12, i64 56, !12, i64 64, !8, i64 72}
!12 = !{!"p1 _ZTS5PgHdr", !13, i64 0}
!13 = !{!"any pointer", !9, i64 0}
!14 = !{!"short", !9, i64 0}
!15 = !{!12, !12, i64 0}
!16 = !{!7, !8, i64 8}
!17 = !{!7, !8, i64 0}
!18 = !{!8, !8, i64 0}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.mustprogress"}
!21 = distinct !{!21, !20}
!22 = !{!23, !23, i64 0}
!23 = !{!"p2 _ZTS5PgHdr", !24, i64 0}
!24 = !{!"any p2 pointer", !13, i64 0}
!25 = distinct !{!25, !20}
!26 = distinct !{!26, !20}
