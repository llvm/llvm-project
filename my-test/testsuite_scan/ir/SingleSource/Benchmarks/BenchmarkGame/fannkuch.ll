; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/fannkuch.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/fannkuch.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [23 x i8] c"Pfannkuchen(%d) = %ld\0A\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call noalias dereferenceable_or_null(44) ptr @calloc(i64 noundef 11, i64 noundef 4) #6
  %4 = tail call noalias dereferenceable_or_null(44) ptr @calloc(i64 noundef 11, i64 noundef 4) #6
  %5 = tail call noalias dereferenceable_or_null(44) ptr @calloc(i64 noundef 11, i64 noundef 4) #6
  %6 = getelementptr nuw i8, ptr %4, i64 4
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 12
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %6, align 4, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 20
  %11 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %12 = getelementptr inbounds nuw i8, ptr %4, i64 28
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %10, align 4, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %4, i64 36
  %15 = getelementptr inbounds nuw i8, ptr %4, i64 40
  store <2 x i32> <i32 9, i32 10>, ptr %14, align 4, !tbaa !6
  %16 = getelementptr nuw i8, ptr %3, i64 4
  %17 = getelementptr inbounds nuw i8, ptr %5, i64 4
  %18 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %19 = getelementptr inbounds nuw i8, ptr %5, i64 12
  %20 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %21 = getelementptr inbounds nuw i8, ptr %5, i64 20
  %22 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %23 = getelementptr inbounds nuw i8, ptr %5, i64 28
  %24 = getelementptr inbounds nuw i8, ptr %5, i64 32
  %25 = getelementptr inbounds nuw i8, ptr %5, i64 36
  %26 = getelementptr inbounds nuw i8, ptr %5, i64 40
  br label %27

27:                                               ; preds = %185, %2
  %28 = phi i32 [ 10, %2 ], [ %186, %185 ]
  %29 = phi i32 [ 10, %2 ], [ %187, %185 ]
  %30 = phi i1 [ false, %2 ], [ %139, %185 ]
  %31 = phi i32 [ 11, %2 ], [ %188, %185 ]
  %32 = phi i32 [ 0, %2 ], [ %71, %185 ]
  %33 = phi i64 [ 0, %2 ], [ %135, %185 ]
  %34 = icmp slt i32 %32, 30
  br i1 %34, label %35, label %70

35:                                               ; preds = %27
  %36 = load i32, ptr %4, align 4, !tbaa !6
  %37 = add nsw i32 %36, 1
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %37)
  %39 = load i32, ptr %6, align 4, !tbaa !6
  %40 = add nsw i32 %39, 1
  %41 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %40)
  %42 = load i32, ptr %7, align 4, !tbaa !6
  %43 = add nsw i32 %42, 1
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %43)
  %45 = load i32, ptr %8, align 4, !tbaa !6
  %46 = add nsw i32 %45, 1
  %47 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %46)
  %48 = load i32, ptr %9, align 4, !tbaa !6
  %49 = add nsw i32 %48, 1
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %49)
  %51 = load i32, ptr %10, align 4, !tbaa !6
  %52 = add nsw i32 %51, 1
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %52)
  %54 = load i32, ptr %11, align 4, !tbaa !6
  %55 = add nsw i32 %54, 1
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %55)
  %57 = load i32, ptr %12, align 4, !tbaa !6
  %58 = add nsw i32 %57, 1
  %59 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %58)
  %60 = load i32, ptr %13, align 4, !tbaa !6
  %61 = add nsw i32 %60, 1
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %61)
  %63 = load i32, ptr %14, align 4, !tbaa !6
  %64 = add nsw i32 %63, 1
  %65 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %64)
  %66 = add nsw i32 %29, 1
  %67 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %66)
  %68 = tail call i32 @putchar(i32 10)
  %69 = add nsw i32 %32, 1
  br label %70

70:                                               ; preds = %35, %27
  %71 = phi i32 [ %69, %35 ], [ %32, %27 ]
  br i1 %30, label %100, label %72

72:                                               ; preds = %70
  %73 = zext nneg i32 %31 to i64
  %74 = and i64 %73, 14
  %75 = add nsw i64 %73, -2
  %76 = insertelement <2 x i64> poison, i64 %75, i64 0
  %77 = shufflevector <2 x i64> %76, <2 x i64> poison, <2 x i32> zeroinitializer
  br label %78

78:                                               ; preds = %97, %72
  %79 = phi i64 [ 0, %72 ], [ %98, %97 ]
  %80 = sub i64 %73, %79
  %81 = trunc i64 %79 to i32
  %82 = sub i32 %31, %81
  %83 = insertelement <2 x i64> poison, i64 %79, i64 0
  %84 = shufflevector <2 x i64> %83, <2 x i64> poison, <2 x i32> zeroinitializer
  %85 = or disjoint <2 x i64> %84, <i64 0, i64 1>
  %86 = icmp ule <2 x i64> %85, %77
  %87 = extractelement <2 x i1> %86, i64 0
  br i1 %87, label %88, label %91

88:                                               ; preds = %78
  %89 = getelementptr i32, ptr %5, i64 %80
  %90 = getelementptr i8, ptr %89, i64 -4
  store i32 %82, ptr %90, align 4, !tbaa !6
  br label %91

91:                                               ; preds = %88, %78
  %92 = extractelement <2 x i1> %86, i64 1
  br i1 %92, label %93, label %97

93:                                               ; preds = %91
  %94 = getelementptr i32, ptr %5, i64 %80
  %95 = getelementptr i8, ptr %94, i64 -8
  %96 = add i32 %82, -1
  store i32 %96, ptr %95, align 4, !tbaa !6
  br label %97

97:                                               ; preds = %93, %91
  %98 = add nuw i64 %79, 2
  %99 = icmp eq i64 %98, %74
  br i1 %99, label %100, label %78, !llvm.loop !10

100:                                              ; preds = %97, %70
  %101 = load i32, ptr %4, align 4, !tbaa !6
  %102 = icmp eq i32 %101, 0
  br i1 %102, label %133, label %103

103:                                              ; preds = %100
  %104 = icmp eq i32 %28, 10
  br i1 %104, label %133, label %105

105:                                              ; preds = %103
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(40) %16, ptr noundef nonnull align 4 dereferenceable(40) %6, i64 40, i1 false), !tbaa !6
  br label %106

106:                                              ; preds = %125, %105
  %107 = phi i32 [ %129, %125 ], [ %101, %105 ]
  %108 = phi i64 [ %127, %125 ], [ 0, %105 ]
  %109 = icmp sgt i32 %107, 2
  br i1 %109, label %112, label %110

110:                                              ; preds = %106
  %111 = sext i32 %107 to i64
  br label %125

112:                                              ; preds = %106
  %113 = zext nneg i32 %107 to i64
  %114 = add nsw i64 %113, -1
  br label %115

115:                                              ; preds = %115, %112
  %116 = phi i64 [ 1, %112 ], [ %122, %115 ]
  %117 = phi i64 [ %114, %112 ], [ %123, %115 ]
  %118 = getelementptr inbounds nuw i32, ptr %3, i64 %116
  %119 = load i32, ptr %118, align 4, !tbaa !6
  %120 = getelementptr inbounds i32, ptr %3, i64 %117
  %121 = load i32, ptr %120, align 4, !tbaa !6
  store i32 %121, ptr %118, align 4, !tbaa !6
  store i32 %119, ptr %120, align 4, !tbaa !6
  %122 = add nuw nsw i64 %116, 1
  %123 = add nsw i64 %117, -1
  %124 = icmp slt i64 %122, %123
  br i1 %124, label %115, label %125, !llvm.loop !14

125:                                              ; preds = %115, %110
  %126 = phi i64 [ %111, %110 ], [ %113, %115 ]
  %127 = add nuw nsw i64 %108, 1
  %128 = getelementptr inbounds i32, ptr %3, i64 %126
  %129 = load i32, ptr %128, align 4, !tbaa !6
  store i32 %107, ptr %128, align 4, !tbaa !6
  %130 = icmp eq i32 %129, 0
  br i1 %130, label %131, label %106, !llvm.loop !15

131:                                              ; preds = %125
  %132 = tail call i64 @llvm.smax.i64(i64 %33, i64 %127)
  br label %133

133:                                              ; preds = %131, %103, %100
  %134 = phi i32 [ %29, %100 ], [ 10, %103 ], [ %28, %131 ]
  %135 = phi i64 [ %33, %100 ], [ %33, %103 ], [ %132, %131 ]
  %136 = load i32, ptr %6, align 4, !tbaa !6
  store i32 %136, ptr %4, align 4, !tbaa !6
  store i32 %101, ptr %6, align 4, !tbaa !6
  %137 = load i32, ptr %17, align 4, !tbaa !6
  %138 = add nsw i32 %137, -1
  store i32 %138, ptr %17, align 4, !tbaa !6
  %139 = icmp sgt i32 %137, 1
  br i1 %139, label %185, label %140

140:                                              ; preds = %133
  %141 = load i64, ptr %6, align 4, !tbaa !6
  store i64 %141, ptr %4, align 4, !tbaa !6
  store i32 %136, ptr %7, align 4, !tbaa !6
  %142 = load i32, ptr %18, align 4, !tbaa !6
  %143 = add nsw i32 %142, -1
  store i32 %143, ptr %18, align 4, !tbaa !6
  %144 = icmp sgt i32 %142, 1
  br i1 %144, label %185, label %145

145:                                              ; preds = %140
  %146 = trunc i64 %141 to i32
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(12) %4, ptr noundef nonnull align 4 dereferenceable(12) %6, i64 12, i1 false), !tbaa !6
  store i32 %146, ptr %8, align 4, !tbaa !6
  %147 = load i32, ptr %19, align 4, !tbaa !6
  %148 = add nsw i32 %147, -1
  store i32 %148, ptr %19, align 4, !tbaa !6
  %149 = icmp sgt i32 %147, 1
  br i1 %149, label %185, label %150

150:                                              ; preds = %145
  %151 = load i32, ptr %4, align 4, !tbaa !6
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %4, ptr noundef nonnull align 4 dereferenceable(16) %6, i64 16, i1 false), !tbaa !6
  store i32 %151, ptr %9, align 4, !tbaa !6
  %152 = load i32, ptr %20, align 4, !tbaa !6
  %153 = add nsw i32 %152, -1
  store i32 %153, ptr %20, align 4, !tbaa !6
  %154 = icmp sgt i32 %152, 1
  br i1 %154, label %185, label %155

155:                                              ; preds = %150
  %156 = load i32, ptr %4, align 4, !tbaa !6
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) %4, ptr noundef nonnull align 4 dereferenceable(20) %6, i64 20, i1 false), !tbaa !6
  store i32 %156, ptr %10, align 4, !tbaa !6
  %157 = load i32, ptr %21, align 4, !tbaa !6
  %158 = add nsw i32 %157, -1
  store i32 %158, ptr %21, align 4, !tbaa !6
  %159 = icmp sgt i32 %157, 1
  br i1 %159, label %185, label %160

160:                                              ; preds = %155
  %161 = load i32, ptr %4, align 4, !tbaa !6
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(24) %4, ptr noundef nonnull align 4 dereferenceable(24) %6, i64 24, i1 false), !tbaa !6
  store i32 %161, ptr %11, align 4, !tbaa !6
  %162 = load i32, ptr %22, align 4, !tbaa !6
  %163 = add nsw i32 %162, -1
  store i32 %163, ptr %22, align 4, !tbaa !6
  %164 = icmp sgt i32 %162, 1
  br i1 %164, label %185, label %165

165:                                              ; preds = %160
  %166 = load i32, ptr %4, align 4, !tbaa !6
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) %4, ptr noundef nonnull align 4 dereferenceable(28) %6, i64 28, i1 false), !tbaa !6
  store i32 %166, ptr %12, align 4, !tbaa !6
  %167 = load i32, ptr %23, align 4, !tbaa !6
  %168 = add nsw i32 %167, -1
  store i32 %168, ptr %23, align 4, !tbaa !6
  %169 = icmp sgt i32 %167, 1
  br i1 %169, label %185, label %170

170:                                              ; preds = %165
  %171 = load i32, ptr %4, align 4, !tbaa !6
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(32) %4, ptr noundef nonnull align 4 dereferenceable(32) %6, i64 32, i1 false), !tbaa !6
  store i32 %171, ptr %13, align 4, !tbaa !6
  %172 = load i32, ptr %24, align 4, !tbaa !6
  %173 = add nsw i32 %172, -1
  store i32 %173, ptr %24, align 4, !tbaa !6
  %174 = icmp sgt i32 %172, 1
  br i1 %174, label %185, label %175

175:                                              ; preds = %170
  %176 = load i32, ptr %4, align 4, !tbaa !6
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(36) %4, ptr noundef nonnull align 4 dereferenceable(36) %6, i64 36, i1 false), !tbaa !6
  store i32 %176, ptr %14, align 4, !tbaa !6
  %177 = load i32, ptr %25, align 4, !tbaa !6
  %178 = add nsw i32 %177, -1
  store i32 %178, ptr %25, align 4, !tbaa !6
  %179 = icmp sgt i32 %177, 1
  br i1 %179, label %185, label %180

180:                                              ; preds = %175
  %181 = load i32, ptr %4, align 4, !tbaa !6
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(40) %4, ptr noundef nonnull align 4 dereferenceable(40) %6, i64 40, i1 false), !tbaa !6
  store i32 %181, ptr %15, align 4, !tbaa !6
  %182 = load i32, ptr %26, align 4, !tbaa !6
  %183 = add nsw i32 %182, -1
  store i32 %183, ptr %26, align 4, !tbaa !6
  %184 = icmp sgt i32 %182, 1
  br i1 %184, label %185, label %189

185:                                              ; preds = %180, %175, %170, %165, %160, %155, %150, %145, %140, %133
  %186 = phi i32 [ %28, %133 ], [ %28, %140 ], [ %28, %145 ], [ %28, %150 ], [ %28, %155 ], [ %28, %160 ], [ %28, %165 ], [ %28, %170 ], [ %28, %175 ], [ %181, %180 ]
  %187 = phi i32 [ %134, %133 ], [ %134, %140 ], [ %134, %145 ], [ %134, %150 ], [ %134, %155 ], [ %134, %160 ], [ %134, %165 ], [ %134, %170 ], [ %134, %175 ], [ %181, %180 ]
  %188 = phi i32 [ 1, %133 ], [ 2, %140 ], [ 3, %145 ], [ 4, %150 ], [ 5, %155 ], [ 6, %160 ], [ 7, %165 ], [ 8, %170 ], [ 9, %175 ], [ 10, %180 ]
  br label %27

189:                                              ; preds = %180
  %190 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 11, i64 noundef %135)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nounwind allocsize(0,1) }

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
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11}
!15 = distinct !{!15, !11}
