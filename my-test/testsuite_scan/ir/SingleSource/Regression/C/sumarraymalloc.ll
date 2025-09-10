; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/sumarraymalloc.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/sumarraymalloc.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [11 x i8] c"Sum1 = %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [11 x i8] c"Sum2 = %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp slt i32 %0, 2
  br i1 %3, label %4, label %6

4:                                                ; preds = %2
  %5 = tail call noalias dereferenceable_or_null(400) ptr @malloc(i64 noundef 400) #6
  br label %15

6:                                                ; preds = %2
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !6
  %9 = tail call i64 @strtol(ptr noundef nonnull captures(none) %8, ptr noundef null, i32 noundef 10) #7
  %10 = shl i64 %9, 32
  %11 = ashr exact i64 %10, 30
  %12 = tail call noalias ptr @malloc(i64 noundef %11) #6
  %13 = and i64 %9, 4294967295
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %41, label %15

15:                                               ; preds = %4, %6
  %16 = phi ptr [ %5, %4 ], [ %12, %6 ]
  %17 = phi i64 [ 100, %4 ], [ %9, %6 ]
  %18 = and i64 %17, 4294967295
  %19 = icmp samesign ult i64 %18, 8
  br i1 %19, label %33, label %20

20:                                               ; preds = %15
  %21 = and i64 %17, 4294967288
  br label %22

22:                                               ; preds = %22, %20
  %23 = phi i64 [ 0, %20 ], [ %28, %22 ]
  %24 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %20 ], [ %29, %22 ]
  %25 = add <4 x i32> %24, splat (i32 4)
  %26 = getelementptr inbounds nuw i32, ptr %16, i64 %23
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16
  store <4 x i32> %24, ptr %26, align 4, !tbaa !11
  store <4 x i32> %25, ptr %27, align 4, !tbaa !11
  %28 = add nuw i64 %23, 8
  %29 = add <4 x i32> %24, splat (i32 8)
  %30 = icmp eq i64 %28, %21
  br i1 %30, label %31, label %22, !llvm.loop !13

31:                                               ; preds = %22
  %32 = icmp eq i64 %18, %21
  br i1 %32, label %41, label %33

33:                                               ; preds = %15, %31
  %34 = phi i64 [ 0, %15 ], [ %21, %31 ]
  br label %35

35:                                               ; preds = %33, %35
  %36 = phi i64 [ %39, %35 ], [ %34, %33 ]
  %37 = getelementptr inbounds nuw i32, ptr %16, i64 %36
  %38 = trunc nuw i64 %36 to i32
  store i32 %38, ptr %37, align 4, !tbaa !11
  %39 = add nuw nsw i64 %36, 1
  %40 = icmp eq i64 %39, %18
  br i1 %40, label %41, label %35, !llvm.loop !17

41:                                               ; preds = %35, %6, %31
  %42 = phi ptr [ %12, %6 ], [ %16, %31 ], [ %16, %35 ]
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 368
  %44 = load <4 x i32>, ptr %43, align 4, !tbaa !11
  %45 = getelementptr inbounds nuw i8, ptr %42, i64 336
  %46 = load <4 x i32>, ptr %45, align 4, !tbaa !11
  %47 = getelementptr inbounds nuw i8, ptr %42, i64 304
  %48 = load <4 x i32>, ptr %47, align 4, !tbaa !11
  %49 = getelementptr inbounds nuw i8, ptr %42, i64 272
  %50 = load <4 x i32>, ptr %49, align 4, !tbaa !11
  %51 = getelementptr inbounds nuw i8, ptr %42, i64 240
  %52 = load <4 x i32>, ptr %51, align 4, !tbaa !11
  %53 = getelementptr inbounds nuw i8, ptr %42, i64 208
  %54 = load <4 x i32>, ptr %53, align 4, !tbaa !11
  %55 = getelementptr inbounds nuw i8, ptr %42, i64 176
  %56 = load <4 x i32>, ptr %55, align 4, !tbaa !11
  %57 = getelementptr inbounds nuw i8, ptr %42, i64 144
  %58 = load <4 x i32>, ptr %57, align 4, !tbaa !11
  %59 = getelementptr inbounds nuw i8, ptr %42, i64 112
  %60 = load <4 x i32>, ptr %59, align 4, !tbaa !11
  %61 = getelementptr inbounds nuw i8, ptr %42, i64 80
  %62 = load <4 x i32>, ptr %61, align 4, !tbaa !11
  %63 = getelementptr inbounds nuw i8, ptr %42, i64 48
  %64 = load <4 x i32>, ptr %63, align 4, !tbaa !11
  %65 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %66 = load <4 x i32>, ptr %65, align 4, !tbaa !11
  %67 = add <4 x i32> %64, %66
  %68 = add <4 x i32> %62, %67
  %69 = add <4 x i32> %60, %68
  %70 = add <4 x i32> %58, %69
  %71 = add <4 x i32> %56, %70
  %72 = add <4 x i32> %54, %71
  %73 = add <4 x i32> %52, %72
  %74 = add <4 x i32> %50, %73
  %75 = add <4 x i32> %48, %74
  %76 = add <4 x i32> %46, %75
  %77 = add <4 x i32> %44, %76
  %78 = getelementptr inbounds nuw i8, ptr %42, i64 352
  %79 = load <4 x i32>, ptr %78, align 4, !tbaa !11
  %80 = getelementptr inbounds nuw i8, ptr %42, i64 320
  %81 = load <4 x i32>, ptr %80, align 4, !tbaa !11
  %82 = getelementptr inbounds nuw i8, ptr %42, i64 288
  %83 = load <4 x i32>, ptr %82, align 4, !tbaa !11
  %84 = getelementptr inbounds nuw i8, ptr %42, i64 256
  %85 = load <4 x i32>, ptr %84, align 4, !tbaa !11
  %86 = getelementptr inbounds nuw i8, ptr %42, i64 224
  %87 = load <4 x i32>, ptr %86, align 4, !tbaa !11
  %88 = getelementptr inbounds nuw i8, ptr %42, i64 192
  %89 = load <4 x i32>, ptr %88, align 4, !tbaa !11
  %90 = getelementptr inbounds nuw i8, ptr %42, i64 160
  %91 = load <4 x i32>, ptr %90, align 4, !tbaa !11
  %92 = getelementptr inbounds nuw i8, ptr %42, i64 128
  %93 = load <4 x i32>, ptr %92, align 4, !tbaa !11
  %94 = getelementptr inbounds nuw i8, ptr %42, i64 96
  %95 = load <4 x i32>, ptr %94, align 4, !tbaa !11
  %96 = getelementptr inbounds nuw i8, ptr %42, i64 64
  %97 = load <4 x i32>, ptr %96, align 4, !tbaa !11
  %98 = getelementptr inbounds nuw i8, ptr %42, i64 32
  %99 = load <4 x i32>, ptr %98, align 4, !tbaa !11
  %100 = load <4 x i32>, ptr %42, align 4, !tbaa !11
  %101 = add <4 x i32> %99, %100
  %102 = add <4 x i32> %97, %101
  %103 = add <4 x i32> %95, %102
  %104 = add <4 x i32> %93, %103
  %105 = add <4 x i32> %91, %104
  %106 = add <4 x i32> %89, %105
  %107 = add <4 x i32> %87, %106
  %108 = add <4 x i32> %85, %107
  %109 = add <4 x i32> %83, %108
  %110 = add <4 x i32> %81, %109
  %111 = add <4 x i32> %79, %110
  %112 = add <4 x i32> %77, %111
  %113 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %112)
  %114 = getelementptr inbounds nuw i8, ptr %42, i64 384
  %115 = load i32, ptr %114, align 4, !tbaa !11
  %116 = add nsw i32 %115, %113
  %117 = getelementptr inbounds nuw i8, ptr %42, i64 388
  %118 = load i32, ptr %117, align 4, !tbaa !11
  %119 = add nsw i32 %118, %116
  %120 = getelementptr inbounds nuw i8, ptr %42, i64 392
  %121 = load i32, ptr %120, align 4, !tbaa !11
  %122 = add nsw i32 %121, %119
  %123 = getelementptr inbounds nuw i8, ptr %42, i64 396
  %124 = load i32, ptr %123, align 4, !tbaa !11
  %125 = add nsw i32 %124, %122
  %126 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %125)
  %127 = getelementptr i8, ptr %42, i64 368
  %128 = load <4 x i32>, ptr %127, align 4, !tbaa !11
  %129 = getelementptr i8, ptr %42, i64 336
  %130 = load <4 x i32>, ptr %129, align 4, !tbaa !11
  %131 = getelementptr i8, ptr %42, i64 304
  %132 = load <4 x i32>, ptr %131, align 4, !tbaa !11
  %133 = getelementptr i8, ptr %42, i64 272
  %134 = load <4 x i32>, ptr %133, align 4, !tbaa !11
  %135 = getelementptr i8, ptr %42, i64 240
  %136 = load <4 x i32>, ptr %135, align 4, !tbaa !11
  %137 = getelementptr i8, ptr %42, i64 208
  %138 = load <4 x i32>, ptr %137, align 4, !tbaa !11
  %139 = getelementptr i8, ptr %42, i64 176
  %140 = load <4 x i32>, ptr %139, align 4, !tbaa !11
  %141 = getelementptr i8, ptr %42, i64 144
  %142 = load <4 x i32>, ptr %141, align 4, !tbaa !11
  %143 = getelementptr i8, ptr %42, i64 112
  %144 = load <4 x i32>, ptr %143, align 4, !tbaa !11
  %145 = getelementptr i8, ptr %42, i64 80
  %146 = load <4 x i32>, ptr %145, align 4, !tbaa !11
  %147 = getelementptr i8, ptr %42, i64 48
  %148 = load <4 x i32>, ptr %147, align 4, !tbaa !11
  %149 = getelementptr i8, ptr %42, i64 16
  %150 = load <4 x i32>, ptr %149, align 4, !tbaa !11
  %151 = add <4 x i32> %148, %150
  %152 = add <4 x i32> %146, %151
  %153 = add <4 x i32> %144, %152
  %154 = add <4 x i32> %142, %153
  %155 = add <4 x i32> %140, %154
  %156 = add <4 x i32> %138, %155
  %157 = add <4 x i32> %136, %156
  %158 = add <4 x i32> %134, %157
  %159 = add <4 x i32> %132, %158
  %160 = add <4 x i32> %130, %159
  %161 = add <4 x i32> %128, %160
  %162 = getelementptr i8, ptr %42, i64 352
  %163 = load <4 x i32>, ptr %162, align 4, !tbaa !11
  %164 = getelementptr i8, ptr %42, i64 320
  %165 = load <4 x i32>, ptr %164, align 4, !tbaa !11
  %166 = getelementptr i8, ptr %42, i64 288
  %167 = load <4 x i32>, ptr %166, align 4, !tbaa !11
  %168 = getelementptr i8, ptr %42, i64 256
  %169 = load <4 x i32>, ptr %168, align 4, !tbaa !11
  %170 = getelementptr i8, ptr %42, i64 224
  %171 = load <4 x i32>, ptr %170, align 4, !tbaa !11
  %172 = getelementptr i8, ptr %42, i64 192
  %173 = load <4 x i32>, ptr %172, align 4, !tbaa !11
  %174 = getelementptr i8, ptr %42, i64 160
  %175 = load <4 x i32>, ptr %174, align 4, !tbaa !11
  %176 = getelementptr i8, ptr %42, i64 128
  %177 = load <4 x i32>, ptr %176, align 4, !tbaa !11
  %178 = getelementptr i8, ptr %42, i64 96
  %179 = load <4 x i32>, ptr %178, align 4, !tbaa !11
  %180 = getelementptr i8, ptr %42, i64 64
  %181 = load <4 x i32>, ptr %180, align 4, !tbaa !11
  %182 = getelementptr i8, ptr %42, i64 32
  %183 = load <4 x i32>, ptr %182, align 4, !tbaa !11
  %184 = load <4 x i32>, ptr %42, align 4, !tbaa !11
  %185 = add <4 x i32> %183, %184
  %186 = add <4 x i32> %181, %185
  %187 = add <4 x i32> %179, %186
  %188 = add <4 x i32> %177, %187
  %189 = add <4 x i32> %175, %188
  %190 = add <4 x i32> %173, %189
  %191 = add <4 x i32> %171, %190
  %192 = add <4 x i32> %169, %191
  %193 = add <4 x i32> %167, %192
  %194 = add <4 x i32> %165, %193
  %195 = add <4 x i32> %163, %194
  %196 = add <4 x i32> %161, %195
  %197 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %196)
  %198 = getelementptr i8, ptr %42, i64 384
  %199 = getelementptr i8, ptr %42, i64 388
  %200 = load i32, ptr %198, align 4, !tbaa !11
  %201 = add nsw i32 %200, %197
  %202 = getelementptr i8, ptr %42, i64 392
  %203 = load i32, ptr %199, align 4, !tbaa !11
  %204 = add nsw i32 %203, %201
  %205 = getelementptr i8, ptr %42, i64 396
  %206 = load i32, ptr %202, align 4, !tbaa !11
  %207 = add nsw i32 %206, %204
  %208 = load i32, ptr %205, align 4, !tbaa !11
  %209 = add nsw i32 %208, %207
  %210 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %209)
  tail call void @free(ptr noundef nonnull %42) #7
  ret i32 0
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #5

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nounwind allocsize(0) }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !16, !15}
