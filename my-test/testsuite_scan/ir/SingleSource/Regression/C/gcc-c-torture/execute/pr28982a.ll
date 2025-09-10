; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr28982a.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr28982a.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@incs = dso_local local_unnamed_addr global [20 x i32] zeroinitializer, align 16
@ptrs = dso_local local_unnamed_addr global [20 x ptr] zeroinitializer, align 8
@results = dso_local local_unnamed_addr global [20 x float] zeroinitializer, align 64
@input = dso_local global [80 x float] zeroinitializer, align 16

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %168, label %3

3:                                                ; preds = %1
  %4 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 152), align 8, !tbaa !6
  %5 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 144), align 8, !tbaa !6
  %6 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 136), align 8, !tbaa !6
  %7 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 128), align 8, !tbaa !6
  %8 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 120), align 8, !tbaa !6
  %9 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 112), align 8, !tbaa !6
  %10 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 104), align 8, !tbaa !6
  %11 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 96), align 8, !tbaa !6
  %12 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 88), align 8, !tbaa !6
  %13 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 80), align 8, !tbaa !6
  %14 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 72), align 8, !tbaa !6
  %15 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 64), align 8, !tbaa !6
  %16 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 56), align 8, !tbaa !6
  %17 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 48), align 8, !tbaa !6
  %18 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 40), align 8, !tbaa !6
  %19 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 32), align 8, !tbaa !6
  %20 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 24), align 8, !tbaa !6
  %21 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 16), align 8, !tbaa !6
  %22 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 8), align 8, !tbaa !6
  %23 = load ptr, ptr @ptrs, align 8, !tbaa !6
  %24 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 76), align 4, !tbaa !11
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 72), align 4, !tbaa !11
  %26 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 68), align 4, !tbaa !11
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 64), align 4, !tbaa !11
  %28 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 60), align 4, !tbaa !11
  %29 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 56), align 4, !tbaa !11
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 52), align 4, !tbaa !11
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 48), align 4, !tbaa !11
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 44), align 4, !tbaa !11
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 40), align 4, !tbaa !11
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 36), align 4, !tbaa !11
  %35 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 32), align 4, !tbaa !11
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 28), align 4, !tbaa !11
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 24), align 4, !tbaa !11
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 20), align 4, !tbaa !11
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 16), align 4, !tbaa !11
  %40 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 12), align 4, !tbaa !11
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 8), align 4, !tbaa !11
  %42 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 4), align 4, !tbaa !11
  %43 = load i32, ptr @incs, align 4, !tbaa !11
  %44 = sext i32 %43 to i64
  %45 = sext i32 %42 to i64
  %46 = sext i32 %41 to i64
  %47 = sext i32 %40 to i64
  %48 = sext i32 %39 to i64
  %49 = sext i32 %38 to i64
  %50 = sext i32 %37 to i64
  %51 = sext i32 %36 to i64
  %52 = sext i32 %35 to i64
  %53 = sext i32 %34 to i64
  %54 = sext i32 %33 to i64
  %55 = sext i32 %32 to i64
  %56 = sext i32 %31 to i64
  %57 = sext i32 %30 to i64
  %58 = sext i32 %29 to i64
  %59 = sext i32 %28 to i64
  %60 = sext i32 %27 to i64
  %61 = sext i32 %26 to i64
  %62 = sext i32 %25 to i64
  %63 = sext i32 %24 to i64
  br label %64

64:                                               ; preds = %3, %64
  %65 = phi float [ 0.000000e+00, %3 ], [ %165, %64 ]
  %66 = phi ptr [ %4, %3 ], [ %166, %64 ]
  %67 = phi float [ 0.000000e+00, %3 ], [ %162, %64 ]
  %68 = phi ptr [ %5, %3 ], [ %163, %64 ]
  %69 = phi float [ 0.000000e+00, %3 ], [ %159, %64 ]
  %70 = phi ptr [ %6, %3 ], [ %160, %64 ]
  %71 = phi float [ 0.000000e+00, %3 ], [ %156, %64 ]
  %72 = phi ptr [ %7, %3 ], [ %157, %64 ]
  %73 = phi float [ 0.000000e+00, %3 ], [ %153, %64 ]
  %74 = phi ptr [ %8, %3 ], [ %154, %64 ]
  %75 = phi float [ 0.000000e+00, %3 ], [ %150, %64 ]
  %76 = phi ptr [ %9, %3 ], [ %151, %64 ]
  %77 = phi float [ 0.000000e+00, %3 ], [ %147, %64 ]
  %78 = phi ptr [ %10, %3 ], [ %148, %64 ]
  %79 = phi float [ 0.000000e+00, %3 ], [ %144, %64 ]
  %80 = phi ptr [ %11, %3 ], [ %145, %64 ]
  %81 = phi float [ 0.000000e+00, %3 ], [ %141, %64 ]
  %82 = phi ptr [ %12, %3 ], [ %142, %64 ]
  %83 = phi float [ 0.000000e+00, %3 ], [ %138, %64 ]
  %84 = phi ptr [ %13, %3 ], [ %139, %64 ]
  %85 = phi float [ 0.000000e+00, %3 ], [ %135, %64 ]
  %86 = phi ptr [ %14, %3 ], [ %136, %64 ]
  %87 = phi float [ 0.000000e+00, %3 ], [ %132, %64 ]
  %88 = phi ptr [ %15, %3 ], [ %133, %64 ]
  %89 = phi float [ 0.000000e+00, %3 ], [ %129, %64 ]
  %90 = phi ptr [ %16, %3 ], [ %130, %64 ]
  %91 = phi float [ 0.000000e+00, %3 ], [ %126, %64 ]
  %92 = phi ptr [ %17, %3 ], [ %127, %64 ]
  %93 = phi float [ 0.000000e+00, %3 ], [ %123, %64 ]
  %94 = phi ptr [ %18, %3 ], [ %124, %64 ]
  %95 = phi float [ 0.000000e+00, %3 ], [ %120, %64 ]
  %96 = phi ptr [ %19, %3 ], [ %121, %64 ]
  %97 = phi float [ 0.000000e+00, %3 ], [ %117, %64 ]
  %98 = phi ptr [ %20, %3 ], [ %118, %64 ]
  %99 = phi float [ 0.000000e+00, %3 ], [ %114, %64 ]
  %100 = phi ptr [ %21, %3 ], [ %115, %64 ]
  %101 = phi float [ 0.000000e+00, %3 ], [ %111, %64 ]
  %102 = phi ptr [ %22, %3 ], [ %112, %64 ]
  %103 = phi float [ 0.000000e+00, %3 ], [ %108, %64 ]
  %104 = phi ptr [ %23, %3 ], [ %109, %64 ]
  %105 = phi i32 [ %0, %3 ], [ %106, %64 ]
  %106 = add nsw i32 %105, -1
  %107 = load float, ptr %104, align 4, !tbaa !13
  %108 = fadd float %103, %107
  %109 = getelementptr inbounds float, ptr %104, i64 %44
  %110 = load float, ptr %102, align 4, !tbaa !13
  %111 = fadd float %101, %110
  %112 = getelementptr inbounds float, ptr %102, i64 %45
  %113 = load float, ptr %100, align 4, !tbaa !13
  %114 = fadd float %99, %113
  %115 = getelementptr inbounds float, ptr %100, i64 %46
  %116 = load float, ptr %98, align 4, !tbaa !13
  %117 = fadd float %97, %116
  %118 = getelementptr inbounds float, ptr %98, i64 %47
  %119 = load float, ptr %96, align 4, !tbaa !13
  %120 = fadd float %95, %119
  %121 = getelementptr inbounds float, ptr %96, i64 %48
  %122 = load float, ptr %94, align 4, !tbaa !13
  %123 = fadd float %93, %122
  %124 = getelementptr inbounds float, ptr %94, i64 %49
  %125 = load float, ptr %92, align 4, !tbaa !13
  %126 = fadd float %91, %125
  %127 = getelementptr inbounds float, ptr %92, i64 %50
  %128 = load float, ptr %90, align 4, !tbaa !13
  %129 = fadd float %89, %128
  %130 = getelementptr inbounds float, ptr %90, i64 %51
  %131 = load float, ptr %88, align 4, !tbaa !13
  %132 = fadd float %87, %131
  %133 = getelementptr inbounds float, ptr %88, i64 %52
  %134 = load float, ptr %86, align 4, !tbaa !13
  %135 = fadd float %85, %134
  %136 = getelementptr inbounds float, ptr %86, i64 %53
  %137 = load float, ptr %84, align 4, !tbaa !13
  %138 = fadd float %83, %137
  %139 = getelementptr inbounds float, ptr %84, i64 %54
  %140 = load float, ptr %82, align 4, !tbaa !13
  %141 = fadd float %81, %140
  %142 = getelementptr inbounds float, ptr %82, i64 %55
  %143 = load float, ptr %80, align 4, !tbaa !13
  %144 = fadd float %79, %143
  %145 = getelementptr inbounds float, ptr %80, i64 %56
  %146 = load float, ptr %78, align 4, !tbaa !13
  %147 = fadd float %77, %146
  %148 = getelementptr inbounds float, ptr %78, i64 %57
  %149 = load float, ptr %76, align 4, !tbaa !13
  %150 = fadd float %75, %149
  %151 = getelementptr inbounds float, ptr %76, i64 %58
  %152 = load float, ptr %74, align 4, !tbaa !13
  %153 = fadd float %73, %152
  %154 = getelementptr inbounds float, ptr %74, i64 %59
  %155 = load float, ptr %72, align 4, !tbaa !13
  %156 = fadd float %71, %155
  %157 = getelementptr inbounds float, ptr %72, i64 %60
  %158 = load float, ptr %70, align 4, !tbaa !13
  %159 = fadd float %69, %158
  %160 = getelementptr inbounds float, ptr %70, i64 %61
  %161 = load float, ptr %68, align 4, !tbaa !13
  %162 = fadd float %67, %161
  %163 = getelementptr inbounds float, ptr %68, i64 %62
  %164 = load float, ptr %66, align 4, !tbaa !13
  %165 = fadd float %65, %164
  %166 = getelementptr inbounds float, ptr %66, i64 %63
  %167 = icmp eq i32 %106, 0
  br i1 %167, label %168, label %64, !llvm.loop !15

168:                                              ; preds = %64, %1
  %169 = phi float [ 0.000000e+00, %1 ], [ %108, %64 ]
  %170 = phi float [ 0.000000e+00, %1 ], [ %111, %64 ]
  %171 = phi float [ 0.000000e+00, %1 ], [ %114, %64 ]
  %172 = phi float [ 0.000000e+00, %1 ], [ %117, %64 ]
  %173 = phi float [ 0.000000e+00, %1 ], [ %120, %64 ]
  %174 = phi float [ 0.000000e+00, %1 ], [ %123, %64 ]
  %175 = phi float [ 0.000000e+00, %1 ], [ %126, %64 ]
  %176 = phi float [ 0.000000e+00, %1 ], [ %129, %64 ]
  %177 = phi float [ 0.000000e+00, %1 ], [ %132, %64 ]
  %178 = phi float [ 0.000000e+00, %1 ], [ %135, %64 ]
  %179 = phi float [ 0.000000e+00, %1 ], [ %138, %64 ]
  %180 = phi float [ 0.000000e+00, %1 ], [ %141, %64 ]
  %181 = phi float [ 0.000000e+00, %1 ], [ %144, %64 ]
  %182 = phi float [ 0.000000e+00, %1 ], [ %147, %64 ]
  %183 = phi float [ 0.000000e+00, %1 ], [ %150, %64 ]
  %184 = phi float [ 0.000000e+00, %1 ], [ %153, %64 ]
  %185 = phi float [ 0.000000e+00, %1 ], [ %156, %64 ]
  %186 = phi float [ 0.000000e+00, %1 ], [ %159, %64 ]
  %187 = phi float [ 0.000000e+00, %1 ], [ %162, %64 ]
  %188 = phi float [ 0.000000e+00, %1 ], [ %165, %64 ]
  store float %169, ptr @results, align 4, !tbaa !13
  store float %170, ptr getelementptr inbounds nuw (i8, ptr @results, i64 4), align 4, !tbaa !13
  store float %171, ptr getelementptr inbounds nuw (i8, ptr @results, i64 8), align 4, !tbaa !13
  store float %172, ptr getelementptr inbounds nuw (i8, ptr @results, i64 12), align 4, !tbaa !13
  store float %173, ptr getelementptr inbounds nuw (i8, ptr @results, i64 16), align 4, !tbaa !13
  store float %174, ptr getelementptr inbounds nuw (i8, ptr @results, i64 20), align 4, !tbaa !13
  store float %175, ptr getelementptr inbounds nuw (i8, ptr @results, i64 24), align 4, !tbaa !13
  store float %176, ptr getelementptr inbounds nuw (i8, ptr @results, i64 28), align 4, !tbaa !13
  store float %177, ptr getelementptr inbounds nuw (i8, ptr @results, i64 32), align 4, !tbaa !13
  store float %178, ptr getelementptr inbounds nuw (i8, ptr @results, i64 36), align 4, !tbaa !13
  store float %179, ptr getelementptr inbounds nuw (i8, ptr @results, i64 40), align 4, !tbaa !13
  store float %180, ptr getelementptr inbounds nuw (i8, ptr @results, i64 44), align 4, !tbaa !13
  store float %181, ptr getelementptr inbounds nuw (i8, ptr @results, i64 48), align 4, !tbaa !13
  store float %182, ptr getelementptr inbounds nuw (i8, ptr @results, i64 52), align 4, !tbaa !13
  store float %183, ptr getelementptr inbounds nuw (i8, ptr @results, i64 56), align 4, !tbaa !13
  store float %184, ptr getelementptr inbounds nuw (i8, ptr @results, i64 60), align 4, !tbaa !13
  store float %185, ptr getelementptr inbounds nuw (i8, ptr @results, i64 64), align 4, !tbaa !13
  store float %186, ptr getelementptr inbounds nuw (i8, ptr @results, i64 68), align 4, !tbaa !13
  store float %187, ptr getelementptr inbounds nuw (i8, ptr @results, i64 72), align 4, !tbaa !13
  store float %188, ptr getelementptr inbounds nuw (i8, ptr @results, i64 76), align 4, !tbaa !13
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #1 {
  store ptr @input, ptr @ptrs, align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 4), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 8), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 8), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 16), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 12), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 24), align 8, !tbaa !6
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr @incs, align 16, !tbaa !11
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 16), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 32), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 20), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 40), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 24), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 48), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 28), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 56), align 8, !tbaa !6
  store <4 x i32> <i32 4, i32 5, i32 6, i32 7>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 16), align 16, !tbaa !11
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 32), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 64), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 36), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 72), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 40), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 80), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 44), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 88), align 8, !tbaa !6
  store <4 x i32> <i32 8, i32 9, i32 10, i32 11>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 32), align 16, !tbaa !11
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 48), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 96), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 52), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 104), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 56), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 112), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 60), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 120), align 8, !tbaa !6
  store <4 x i32> <i32 12, i32 13, i32 14, i32 15>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 48), align 16, !tbaa !11
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 64), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 128), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 68), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 136), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 72), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 144), align 8, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 76), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 152), align 8, !tbaa !6
  store <4 x i32> <i32 16, i32 17, i32 18, i32 19>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 64), align 16, !tbaa !11
  store <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, ptr @input, align 16, !tbaa !13
  store <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 16), align 16, !tbaa !13
  store <4 x float> <float 8.000000e+00, float 9.000000e+00, float 1.000000e+01, float 1.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 32), align 16, !tbaa !13
  store <4 x float> <float 1.200000e+01, float 1.300000e+01, float 1.400000e+01, float 1.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 48), align 16, !tbaa !13
  store <4 x float> <float 1.600000e+01, float 1.700000e+01, float 1.800000e+01, float 1.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 64), align 16, !tbaa !13
  store <4 x float> <float 2.000000e+01, float 2.100000e+01, float 2.200000e+01, float 2.300000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 80), align 16, !tbaa !13
  store <4 x float> <float 2.400000e+01, float 2.500000e+01, float 2.600000e+01, float 2.700000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 96), align 16, !tbaa !13
  store <4 x float> <float 2.800000e+01, float 2.900000e+01, float 3.000000e+01, float 3.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 112), align 16, !tbaa !13
  store <4 x float> <float 3.200000e+01, float 3.300000e+01, float 3.400000e+01, float 3.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 128), align 16, !tbaa !13
  store <4 x float> <float 3.600000e+01, float 3.700000e+01, float 3.800000e+01, float 3.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 144), align 16, !tbaa !13
  store <4 x float> <float 4.000000e+01, float 4.100000e+01, float 4.200000e+01, float 4.300000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 160), align 16, !tbaa !13
  store <4 x float> <float 4.400000e+01, float 4.500000e+01, float 4.600000e+01, float 4.700000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 176), align 16, !tbaa !13
  store <4 x float> <float 4.800000e+01, float 4.900000e+01, float 5.000000e+01, float 5.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 192), align 16, !tbaa !13
  store <4 x float> <float 5.200000e+01, float 5.300000e+01, float 5.400000e+01, float 5.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 208), align 16, !tbaa !13
  store <4 x float> <float 5.600000e+01, float 5.700000e+01, float 5.800000e+01, float 5.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 224), align 16, !tbaa !13
  store <4 x float> <float 6.000000e+01, float 6.100000e+01, float 6.200000e+01, float 6.300000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 240), align 16, !tbaa !13
  store <4 x float> <float 6.400000e+01, float 6.500000e+01, float 6.600000e+01, float 6.700000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 256), align 16, !tbaa !13
  store <4 x float> <float 6.800000e+01, float 6.900000e+01, float 7.000000e+01, float 7.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 272), align 16, !tbaa !13
  store <4 x float> <float 7.200000e+01, float 7.300000e+01, float 7.400000e+01, float 7.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 288), align 16, !tbaa !13
  store <4 x float> <float 7.600000e+01, float 7.700000e+01, float 7.800000e+01, float 7.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 304), align 16, !tbaa !13
  tail call void @foo(i32 noundef 4)
  %1 = load <16 x float>, ptr @results, align 64
  %2 = freeze <16 x float> %1
  %3 = fcmp une <16 x float> %2, <float 0.000000e+00, float 1.000000e+01, float 2.000000e+01, float 3.000000e+01, float 4.000000e+01, float 5.000000e+01, float 6.000000e+01, float 7.000000e+01, float 8.000000e+01, float 9.000000e+01, float 1.000000e+02, float 1.100000e+02, float 1.200000e+02, float 1.300000e+02, float 1.400000e+02, float 1.500000e+02>
  %4 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @results, i64 64), align 64
  %5 = freeze <4 x float> %4
  %6 = fcmp une <4 x float> %5, <float 1.600000e+02, float 1.700000e+02, float 1.800000e+02, float 1.900000e+02>
  %7 = shufflevector <16 x i1> %3, <16 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %8 = or <4 x i1> %7, %6
  %9 = shufflevector <4 x i1> %8, <4 x i1> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %10 = shufflevector <16 x i1> %9, <16 x i1> %3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = bitcast <16 x i1> %10 to i16
  %12 = icmp ne i16 %11, 0
  %13 = zext i1 %12 to i32
  ret i32 %13
}

attributes #0 = { nofree noinline norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 float", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"float", !9, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
