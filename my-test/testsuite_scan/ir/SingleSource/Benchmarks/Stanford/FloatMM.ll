; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/FloatMM.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/FloatMM.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.element = type { i32, i32 }
%struct.complex = type { float, float }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@rma = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmb = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmr = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@value = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@fixed = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@floated = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@permarray = dso_local local_unnamed_addr global [11 x i32] zeroinitializer, align 4
@pctr = dso_local local_unnamed_addr global i32 0, align 4
@tree = dso_local local_unnamed_addr global ptr null, align 8
@stack = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@cellspace = dso_local local_unnamed_addr global [19 x %struct.element] zeroinitializer, align 4
@freelist = dso_local local_unnamed_addr global i32 0, align 4
@movesdone = dso_local local_unnamed_addr global i32 0, align 4
@ima = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imb = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imr = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@piececount = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@class = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@piecemax = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@puzzl = dso_local local_unnamed_addr global [512 x i32] zeroinitializer, align 4
@p = dso_local local_unnamed_addr global [13 x [512 x i32]] zeroinitializer, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@kount = dso_local local_unnamed_addr global i32 0, align 4
@sortlist = dso_local local_unnamed_addr global [5001 x i32] zeroinitializer, align 4
@biggest = dso_local local_unnamed_addr global i32 0, align 4
@littlest = dso_local local_unnamed_addr global i32 0, align 4
@top = dso_local local_unnamed_addr global i32 0, align 4
@z = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@w = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@e = dso_local local_unnamed_addr global [130 x %struct.complex] zeroinitializer, align 4
@zr = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@zi = dso_local local_unnamed_addr global float 0.000000e+00, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Initrand() local_unnamed_addr #0 {
  store i64 74755, ptr @seed, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 65536) i32 @Rand() local_unnamed_addr #1 {
  %1 = load i64, ptr @seed, align 8, !tbaa !6
  %2 = mul nsw i64 %1, 1309
  %3 = add nsw i64 %2, 13849
  %4 = and i64 %3, 65535
  store i64 %4, ptr @seed, align 8, !tbaa !6
  %5 = trunc nuw nsw i64 %4 to i32
  ret i32 %5
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @rInitmatrix(ptr noundef writeonly captures(none) %0) local_unnamed_addr #2 {
  %2 = load i64, ptr @seed, align 8, !tbaa !6
  %3 = freeze i64 %2
  br label %4

4:                                                ; preds = %1, %23
  %5 = phi i64 [ 1, %1 ], [ %24, %23 ]
  %6 = phi i64 [ %3, %1 ], [ %13, %23 ]
  %7 = getelementptr inbounds nuw [41 x float], ptr %0, i64 %5
  br label %8

8:                                                ; preds = %4, %8
  %9 = phi i64 [ 1, %4 ], [ %21, %8 ]
  %10 = phi i64 [ %6, %4 ], [ %13, %8 ]
  %11 = mul i64 %10, 1309
  %12 = add i64 %11, 13849
  %13 = and i64 %12, 65535
  %14 = trunc i64 %12 to i16
  %15 = urem i16 %14, 120
  %16 = zext nneg i16 %15 to i32
  %17 = add nsw i32 %16, -60
  %18 = sitofp i32 %17 to float
  %19 = fdiv float %18, 3.000000e+00
  %20 = getelementptr inbounds nuw float, ptr %7, i64 %9
  store float %19, ptr %20, align 4, !tbaa !10
  %21 = add nuw nsw i64 %9, 1
  %22 = icmp eq i64 %21, 41
  br i1 %22, label %23, label %8, !llvm.loop !12

23:                                               ; preds = %8
  %24 = add nuw nsw i64 %5, 1
  %25 = icmp eq i64 %24, 41
  br i1 %25, label %26, label %4, !llvm.loop !14

26:                                               ; preds = %23
  store i64 %13, ptr @seed, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @rInnerproduct(ptr noundef writeonly captures(none) initializes((0, 4)) %0, ptr noundef readonly captures(none) %1, ptr noundef readonly captures(none) %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #3 {
  store float 0.000000e+00, ptr %0, align 4, !tbaa !10
  %6 = sext i32 %3 to i64
  %7 = getelementptr inbounds [41 x float], ptr %1, i64 %6
  %8 = sext i32 %4 to i64
  %9 = getelementptr float, ptr %2, i64 %8
  %10 = getelementptr inbounds nuw i8, ptr %7, i64 4
  %11 = load float, ptr %10, align 4, !tbaa !10
  %12 = getelementptr i8, ptr %9, i64 164
  %13 = load float, ptr %12, align 4, !tbaa !10
  %14 = tail call float @llvm.fmuladd.f32(float %11, float %13, float 0.000000e+00)
  store float %14, ptr %0, align 4, !tbaa !10
  %15 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %16 = load float, ptr %15, align 4, !tbaa !10
  %17 = getelementptr i8, ptr %9, i64 328
  %18 = load float, ptr %17, align 4, !tbaa !10
  %19 = tail call float @llvm.fmuladd.f32(float %16, float %18, float %14)
  store float %19, ptr %0, align 4, !tbaa !10
  %20 = getelementptr inbounds nuw i8, ptr %7, i64 12
  %21 = load float, ptr %20, align 4, !tbaa !10
  %22 = getelementptr i8, ptr %9, i64 492
  %23 = load float, ptr %22, align 4, !tbaa !10
  %24 = tail call float @llvm.fmuladd.f32(float %21, float %23, float %19)
  store float %24, ptr %0, align 4, !tbaa !10
  %25 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %26 = load float, ptr %25, align 4, !tbaa !10
  %27 = getelementptr i8, ptr %9, i64 656
  %28 = load float, ptr %27, align 4, !tbaa !10
  %29 = tail call float @llvm.fmuladd.f32(float %26, float %28, float %24)
  store float %29, ptr %0, align 4, !tbaa !10
  %30 = getelementptr inbounds nuw i8, ptr %7, i64 20
  %31 = load float, ptr %30, align 4, !tbaa !10
  %32 = getelementptr i8, ptr %9, i64 820
  %33 = load float, ptr %32, align 4, !tbaa !10
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  store float %34, ptr %0, align 4, !tbaa !10
  %35 = getelementptr inbounds nuw i8, ptr %7, i64 24
  %36 = load float, ptr %35, align 4, !tbaa !10
  %37 = getelementptr i8, ptr %9, i64 984
  %38 = load float, ptr %37, align 4, !tbaa !10
  %39 = tail call float @llvm.fmuladd.f32(float %36, float %38, float %34)
  store float %39, ptr %0, align 4, !tbaa !10
  %40 = getelementptr inbounds nuw i8, ptr %7, i64 28
  %41 = load float, ptr %40, align 4, !tbaa !10
  %42 = getelementptr i8, ptr %9, i64 1148
  %43 = load float, ptr %42, align 4, !tbaa !10
  %44 = tail call float @llvm.fmuladd.f32(float %41, float %43, float %39)
  store float %44, ptr %0, align 4, !tbaa !10
  %45 = getelementptr inbounds nuw i8, ptr %7, i64 32
  %46 = load float, ptr %45, align 4, !tbaa !10
  %47 = getelementptr i8, ptr %9, i64 1312
  %48 = load float, ptr %47, align 4, !tbaa !10
  %49 = tail call float @llvm.fmuladd.f32(float %46, float %48, float %44)
  store float %49, ptr %0, align 4, !tbaa !10
  %50 = getelementptr inbounds nuw i8, ptr %7, i64 36
  %51 = load float, ptr %50, align 4, !tbaa !10
  %52 = getelementptr i8, ptr %9, i64 1476
  %53 = load float, ptr %52, align 4, !tbaa !10
  %54 = tail call float @llvm.fmuladd.f32(float %51, float %53, float %49)
  store float %54, ptr %0, align 4, !tbaa !10
  %55 = getelementptr inbounds nuw i8, ptr %7, i64 40
  %56 = load float, ptr %55, align 4, !tbaa !10
  %57 = getelementptr i8, ptr %9, i64 1640
  %58 = load float, ptr %57, align 4, !tbaa !10
  %59 = tail call float @llvm.fmuladd.f32(float %56, float %58, float %54)
  store float %59, ptr %0, align 4, !tbaa !10
  %60 = getelementptr inbounds nuw i8, ptr %7, i64 44
  %61 = load float, ptr %60, align 4, !tbaa !10
  %62 = getelementptr i8, ptr %9, i64 1804
  %63 = load float, ptr %62, align 4, !tbaa !10
  %64 = tail call float @llvm.fmuladd.f32(float %61, float %63, float %59)
  store float %64, ptr %0, align 4, !tbaa !10
  %65 = getelementptr inbounds nuw i8, ptr %7, i64 48
  %66 = load float, ptr %65, align 4, !tbaa !10
  %67 = getelementptr i8, ptr %9, i64 1968
  %68 = load float, ptr %67, align 4, !tbaa !10
  %69 = tail call float @llvm.fmuladd.f32(float %66, float %68, float %64)
  store float %69, ptr %0, align 4, !tbaa !10
  %70 = getelementptr inbounds nuw i8, ptr %7, i64 52
  %71 = load float, ptr %70, align 4, !tbaa !10
  %72 = getelementptr i8, ptr %9, i64 2132
  %73 = load float, ptr %72, align 4, !tbaa !10
  %74 = tail call float @llvm.fmuladd.f32(float %71, float %73, float %69)
  store float %74, ptr %0, align 4, !tbaa !10
  %75 = getelementptr inbounds nuw i8, ptr %7, i64 56
  %76 = load float, ptr %75, align 4, !tbaa !10
  %77 = getelementptr i8, ptr %9, i64 2296
  %78 = load float, ptr %77, align 4, !tbaa !10
  %79 = tail call float @llvm.fmuladd.f32(float %76, float %78, float %74)
  store float %79, ptr %0, align 4, !tbaa !10
  %80 = getelementptr inbounds nuw i8, ptr %7, i64 60
  %81 = load float, ptr %80, align 4, !tbaa !10
  %82 = getelementptr i8, ptr %9, i64 2460
  %83 = load float, ptr %82, align 4, !tbaa !10
  %84 = tail call float @llvm.fmuladd.f32(float %81, float %83, float %79)
  store float %84, ptr %0, align 4, !tbaa !10
  %85 = getelementptr inbounds nuw i8, ptr %7, i64 64
  %86 = load float, ptr %85, align 4, !tbaa !10
  %87 = getelementptr i8, ptr %9, i64 2624
  %88 = load float, ptr %87, align 4, !tbaa !10
  %89 = tail call float @llvm.fmuladd.f32(float %86, float %88, float %84)
  store float %89, ptr %0, align 4, !tbaa !10
  %90 = getelementptr inbounds nuw i8, ptr %7, i64 68
  %91 = load float, ptr %90, align 4, !tbaa !10
  %92 = getelementptr i8, ptr %9, i64 2788
  %93 = load float, ptr %92, align 4, !tbaa !10
  %94 = tail call float @llvm.fmuladd.f32(float %91, float %93, float %89)
  store float %94, ptr %0, align 4, !tbaa !10
  %95 = getelementptr inbounds nuw i8, ptr %7, i64 72
  %96 = load float, ptr %95, align 4, !tbaa !10
  %97 = getelementptr i8, ptr %9, i64 2952
  %98 = load float, ptr %97, align 4, !tbaa !10
  %99 = tail call float @llvm.fmuladd.f32(float %96, float %98, float %94)
  store float %99, ptr %0, align 4, !tbaa !10
  %100 = getelementptr inbounds nuw i8, ptr %7, i64 76
  %101 = load float, ptr %100, align 4, !tbaa !10
  %102 = getelementptr i8, ptr %9, i64 3116
  %103 = load float, ptr %102, align 4, !tbaa !10
  %104 = tail call float @llvm.fmuladd.f32(float %101, float %103, float %99)
  store float %104, ptr %0, align 4, !tbaa !10
  %105 = getelementptr inbounds nuw i8, ptr %7, i64 80
  %106 = load float, ptr %105, align 4, !tbaa !10
  %107 = getelementptr i8, ptr %9, i64 3280
  %108 = load float, ptr %107, align 4, !tbaa !10
  %109 = tail call float @llvm.fmuladd.f32(float %106, float %108, float %104)
  store float %109, ptr %0, align 4, !tbaa !10
  %110 = getelementptr inbounds nuw i8, ptr %7, i64 84
  %111 = load float, ptr %110, align 4, !tbaa !10
  %112 = getelementptr i8, ptr %9, i64 3444
  %113 = load float, ptr %112, align 4, !tbaa !10
  %114 = tail call float @llvm.fmuladd.f32(float %111, float %113, float %109)
  store float %114, ptr %0, align 4, !tbaa !10
  %115 = getelementptr inbounds nuw i8, ptr %7, i64 88
  %116 = load float, ptr %115, align 4, !tbaa !10
  %117 = getelementptr i8, ptr %9, i64 3608
  %118 = load float, ptr %117, align 4, !tbaa !10
  %119 = tail call float @llvm.fmuladd.f32(float %116, float %118, float %114)
  store float %119, ptr %0, align 4, !tbaa !10
  %120 = getelementptr inbounds nuw i8, ptr %7, i64 92
  %121 = load float, ptr %120, align 4, !tbaa !10
  %122 = getelementptr i8, ptr %9, i64 3772
  %123 = load float, ptr %122, align 4, !tbaa !10
  %124 = tail call float @llvm.fmuladd.f32(float %121, float %123, float %119)
  store float %124, ptr %0, align 4, !tbaa !10
  %125 = getelementptr inbounds nuw i8, ptr %7, i64 96
  %126 = load float, ptr %125, align 4, !tbaa !10
  %127 = getelementptr i8, ptr %9, i64 3936
  %128 = load float, ptr %127, align 4, !tbaa !10
  %129 = tail call float @llvm.fmuladd.f32(float %126, float %128, float %124)
  store float %129, ptr %0, align 4, !tbaa !10
  %130 = getelementptr inbounds nuw i8, ptr %7, i64 100
  %131 = load float, ptr %130, align 4, !tbaa !10
  %132 = getelementptr i8, ptr %9, i64 4100
  %133 = load float, ptr %132, align 4, !tbaa !10
  %134 = tail call float @llvm.fmuladd.f32(float %131, float %133, float %129)
  store float %134, ptr %0, align 4, !tbaa !10
  %135 = getelementptr inbounds nuw i8, ptr %7, i64 104
  %136 = load float, ptr %135, align 4, !tbaa !10
  %137 = getelementptr i8, ptr %9, i64 4264
  %138 = load float, ptr %137, align 4, !tbaa !10
  %139 = tail call float @llvm.fmuladd.f32(float %136, float %138, float %134)
  store float %139, ptr %0, align 4, !tbaa !10
  %140 = getelementptr inbounds nuw i8, ptr %7, i64 108
  %141 = load float, ptr %140, align 4, !tbaa !10
  %142 = getelementptr i8, ptr %9, i64 4428
  %143 = load float, ptr %142, align 4, !tbaa !10
  %144 = tail call float @llvm.fmuladd.f32(float %141, float %143, float %139)
  store float %144, ptr %0, align 4, !tbaa !10
  %145 = getelementptr inbounds nuw i8, ptr %7, i64 112
  %146 = load float, ptr %145, align 4, !tbaa !10
  %147 = getelementptr i8, ptr %9, i64 4592
  %148 = load float, ptr %147, align 4, !tbaa !10
  %149 = tail call float @llvm.fmuladd.f32(float %146, float %148, float %144)
  store float %149, ptr %0, align 4, !tbaa !10
  %150 = getelementptr inbounds nuw i8, ptr %7, i64 116
  %151 = load float, ptr %150, align 4, !tbaa !10
  %152 = getelementptr i8, ptr %9, i64 4756
  %153 = load float, ptr %152, align 4, !tbaa !10
  %154 = tail call float @llvm.fmuladd.f32(float %151, float %153, float %149)
  store float %154, ptr %0, align 4, !tbaa !10
  %155 = getelementptr inbounds nuw i8, ptr %7, i64 120
  %156 = load float, ptr %155, align 4, !tbaa !10
  %157 = getelementptr i8, ptr %9, i64 4920
  %158 = load float, ptr %157, align 4, !tbaa !10
  %159 = tail call float @llvm.fmuladd.f32(float %156, float %158, float %154)
  store float %159, ptr %0, align 4, !tbaa !10
  %160 = getelementptr inbounds nuw i8, ptr %7, i64 124
  %161 = load float, ptr %160, align 4, !tbaa !10
  %162 = getelementptr i8, ptr %9, i64 5084
  %163 = load float, ptr %162, align 4, !tbaa !10
  %164 = tail call float @llvm.fmuladd.f32(float %161, float %163, float %159)
  store float %164, ptr %0, align 4, !tbaa !10
  %165 = getelementptr inbounds nuw i8, ptr %7, i64 128
  %166 = load float, ptr %165, align 4, !tbaa !10
  %167 = getelementptr i8, ptr %9, i64 5248
  %168 = load float, ptr %167, align 4, !tbaa !10
  %169 = tail call float @llvm.fmuladd.f32(float %166, float %168, float %164)
  store float %169, ptr %0, align 4, !tbaa !10
  %170 = getelementptr inbounds nuw i8, ptr %7, i64 132
  %171 = load float, ptr %170, align 4, !tbaa !10
  %172 = getelementptr i8, ptr %9, i64 5412
  %173 = load float, ptr %172, align 4, !tbaa !10
  %174 = tail call float @llvm.fmuladd.f32(float %171, float %173, float %169)
  store float %174, ptr %0, align 4, !tbaa !10
  %175 = getelementptr inbounds nuw i8, ptr %7, i64 136
  %176 = load float, ptr %175, align 4, !tbaa !10
  %177 = getelementptr i8, ptr %9, i64 5576
  %178 = load float, ptr %177, align 4, !tbaa !10
  %179 = tail call float @llvm.fmuladd.f32(float %176, float %178, float %174)
  store float %179, ptr %0, align 4, !tbaa !10
  %180 = getelementptr inbounds nuw i8, ptr %7, i64 140
  %181 = load float, ptr %180, align 4, !tbaa !10
  %182 = getelementptr i8, ptr %9, i64 5740
  %183 = load float, ptr %182, align 4, !tbaa !10
  %184 = tail call float @llvm.fmuladd.f32(float %181, float %183, float %179)
  store float %184, ptr %0, align 4, !tbaa !10
  %185 = getelementptr inbounds nuw i8, ptr %7, i64 144
  %186 = load float, ptr %185, align 4, !tbaa !10
  %187 = getelementptr i8, ptr %9, i64 5904
  %188 = load float, ptr %187, align 4, !tbaa !10
  %189 = tail call float @llvm.fmuladd.f32(float %186, float %188, float %184)
  store float %189, ptr %0, align 4, !tbaa !10
  %190 = getelementptr inbounds nuw i8, ptr %7, i64 148
  %191 = load float, ptr %190, align 4, !tbaa !10
  %192 = getelementptr i8, ptr %9, i64 6068
  %193 = load float, ptr %192, align 4, !tbaa !10
  %194 = tail call float @llvm.fmuladd.f32(float %191, float %193, float %189)
  store float %194, ptr %0, align 4, !tbaa !10
  %195 = getelementptr inbounds nuw i8, ptr %7, i64 152
  %196 = load float, ptr %195, align 4, !tbaa !10
  %197 = getelementptr i8, ptr %9, i64 6232
  %198 = load float, ptr %197, align 4, !tbaa !10
  %199 = tail call float @llvm.fmuladd.f32(float %196, float %198, float %194)
  store float %199, ptr %0, align 4, !tbaa !10
  %200 = getelementptr inbounds nuw i8, ptr %7, i64 156
  %201 = load float, ptr %200, align 4, !tbaa !10
  %202 = getelementptr i8, ptr %9, i64 6396
  %203 = load float, ptr %202, align 4, !tbaa !10
  %204 = tail call float @llvm.fmuladd.f32(float %201, float %203, float %199)
  store float %204, ptr %0, align 4, !tbaa !10
  %205 = getelementptr inbounds nuw i8, ptr %7, i64 160
  %206 = load float, ptr %205, align 4, !tbaa !10
  %207 = getelementptr i8, ptr %9, i64 6560
  %208 = load float, ptr %207, align 4, !tbaa !10
  %209 = tail call float @llvm.fmuladd.f32(float %206, float %208, float %204)
  store float %209, ptr %0, align 4, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #4

; Function Attrs: nofree nounwind uwtable
define dso_local void @Mm(i32 noundef %0) local_unnamed_addr #5 {
  br label %2

2:                                                ; preds = %21, %1
  %3 = phi i64 [ 1, %1 ], [ %22, %21 ]
  %4 = phi i64 [ 74755, %1 ], [ %11, %21 ]
  %5 = getelementptr inbounds nuw [41 x float], ptr @rma, i64 %3
  br label %6

6:                                                ; preds = %6, %2
  %7 = phi i64 [ 1, %2 ], [ %19, %6 ]
  %8 = phi i64 [ %4, %2 ], [ %11, %6 ]
  %9 = mul nuw nsw i64 %8, 1309
  %10 = add nuw nsw i64 %9, 13849
  %11 = and i64 %10, 65535
  %12 = trunc i64 %10 to i16
  %13 = urem i16 %12, 120
  %14 = zext nneg i16 %13 to i32
  %15 = add nsw i32 %14, -60
  %16 = sitofp i32 %15 to float
  %17 = fdiv float %16, 3.000000e+00
  %18 = getelementptr inbounds nuw float, ptr %5, i64 %7
  store float %17, ptr %18, align 4, !tbaa !10
  %19 = add nuw nsw i64 %7, 1
  %20 = icmp eq i64 %19, 41
  br i1 %20, label %21, label %6, !llvm.loop !12

21:                                               ; preds = %6
  %22 = add nuw nsw i64 %3, 1
  %23 = icmp eq i64 %22, 41
  br i1 %23, label %24, label %2, !llvm.loop !14

24:                                               ; preds = %21, %43
  %25 = phi i64 [ %44, %43 ], [ 1, %21 ]
  %26 = phi i64 [ %33, %43 ], [ %11, %21 ]
  %27 = getelementptr inbounds nuw [41 x float], ptr @rmb, i64 %25
  br label %28

28:                                               ; preds = %28, %24
  %29 = phi i64 [ 1, %24 ], [ %41, %28 ]
  %30 = phi i64 [ %26, %24 ], [ %33, %28 ]
  %31 = mul nuw nsw i64 %30, 1309
  %32 = add nuw nsw i64 %31, 13849
  %33 = and i64 %32, 65535
  %34 = trunc i64 %32 to i16
  %35 = urem i16 %34, 120
  %36 = zext nneg i16 %35 to i32
  %37 = add nsw i32 %36, -60
  %38 = sitofp i32 %37 to float
  %39 = fdiv float %38, 3.000000e+00
  %40 = getelementptr inbounds nuw float, ptr %27, i64 %29
  store float %39, ptr %40, align 4, !tbaa !10
  %41 = add nuw nsw i64 %29, 1
  %42 = icmp eq i64 %41, 41
  br i1 %42, label %43, label %28, !llvm.loop !12

43:                                               ; preds = %28
  %44 = add nuw nsw i64 %25, 1
  %45 = icmp eq i64 %44, 41
  br i1 %45, label %46, label %24, !llvm.loop !14

46:                                               ; preds = %43
  store i64 %33, ptr @seed, align 8, !tbaa !6
  br label %47

47:                                               ; preds = %46, %338
  %48 = phi i64 [ 1, %46 ], [ %339, %338 ]
  %49 = getelementptr inbounds nuw [41 x float], ptr @rmr, i64 %48
  %50 = getelementptr inbounds nuw [41 x float], ptr @rma, i64 %48
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 160
  %52 = load float, ptr %51, align 4, !tbaa !10
  %53 = getelementptr inbounds nuw i8, ptr %50, i64 156
  %54 = load float, ptr %53, align 4, !tbaa !10
  %55 = getelementptr inbounds nuw i8, ptr %50, i64 152
  %56 = load float, ptr %55, align 4, !tbaa !10
  %57 = getelementptr inbounds nuw i8, ptr %50, i64 148
  %58 = load float, ptr %57, align 4, !tbaa !10
  %59 = getelementptr inbounds nuw i8, ptr %50, i64 144
  %60 = load float, ptr %59, align 4, !tbaa !10
  %61 = getelementptr inbounds nuw i8, ptr %50, i64 140
  %62 = load float, ptr %61, align 4, !tbaa !10
  %63 = getelementptr inbounds nuw i8, ptr %50, i64 136
  %64 = load float, ptr %63, align 4, !tbaa !10
  %65 = getelementptr inbounds nuw i8, ptr %50, i64 132
  %66 = load float, ptr %65, align 4, !tbaa !10
  %67 = getelementptr inbounds nuw i8, ptr %50, i64 128
  %68 = load float, ptr %67, align 4, !tbaa !10
  %69 = getelementptr inbounds nuw i8, ptr %50, i64 124
  %70 = load float, ptr %69, align 4, !tbaa !10
  %71 = getelementptr inbounds nuw i8, ptr %50, i64 120
  %72 = load float, ptr %71, align 4, !tbaa !10
  %73 = getelementptr inbounds nuw i8, ptr %50, i64 116
  %74 = load float, ptr %73, align 4, !tbaa !10
  %75 = getelementptr inbounds nuw i8, ptr %50, i64 112
  %76 = load float, ptr %75, align 4, !tbaa !10
  %77 = getelementptr inbounds nuw i8, ptr %50, i64 108
  %78 = load float, ptr %77, align 4, !tbaa !10
  %79 = getelementptr inbounds nuw i8, ptr %50, i64 104
  %80 = load float, ptr %79, align 4, !tbaa !10
  %81 = getelementptr inbounds nuw i8, ptr %50, i64 100
  %82 = load float, ptr %81, align 4, !tbaa !10
  %83 = getelementptr inbounds nuw i8, ptr %50, i64 96
  %84 = load float, ptr %83, align 4, !tbaa !10
  %85 = getelementptr inbounds nuw i8, ptr %50, i64 92
  %86 = load float, ptr %85, align 4, !tbaa !10
  %87 = getelementptr inbounds nuw i8, ptr %50, i64 88
  %88 = load float, ptr %87, align 4, !tbaa !10
  %89 = getelementptr inbounds nuw i8, ptr %50, i64 84
  %90 = load float, ptr %89, align 4, !tbaa !10
  %91 = getelementptr inbounds nuw i8, ptr %50, i64 80
  %92 = load float, ptr %91, align 4, !tbaa !10
  %93 = getelementptr inbounds nuw i8, ptr %50, i64 76
  %94 = load float, ptr %93, align 4, !tbaa !10
  %95 = getelementptr inbounds nuw i8, ptr %50, i64 72
  %96 = load float, ptr %95, align 4, !tbaa !10
  %97 = getelementptr inbounds nuw i8, ptr %50, i64 68
  %98 = load float, ptr %97, align 4, !tbaa !10
  %99 = getelementptr inbounds nuw i8, ptr %50, i64 64
  %100 = load float, ptr %99, align 4, !tbaa !10
  %101 = getelementptr inbounds nuw i8, ptr %50, i64 60
  %102 = load float, ptr %101, align 4, !tbaa !10
  %103 = getelementptr inbounds nuw i8, ptr %50, i64 56
  %104 = load float, ptr %103, align 4, !tbaa !10
  %105 = getelementptr inbounds nuw i8, ptr %50, i64 52
  %106 = load float, ptr %105, align 4, !tbaa !10
  %107 = getelementptr inbounds nuw i8, ptr %50, i64 48
  %108 = load float, ptr %107, align 4, !tbaa !10
  %109 = getelementptr inbounds nuw i8, ptr %50, i64 44
  %110 = load float, ptr %109, align 4, !tbaa !10
  %111 = getelementptr inbounds nuw i8, ptr %50, i64 40
  %112 = load float, ptr %111, align 4, !tbaa !10
  %113 = getelementptr inbounds nuw i8, ptr %50, i64 36
  %114 = load float, ptr %113, align 4, !tbaa !10
  %115 = getelementptr inbounds nuw i8, ptr %50, i64 32
  %116 = load float, ptr %115, align 4, !tbaa !10
  %117 = getelementptr inbounds nuw i8, ptr %50, i64 28
  %118 = load float, ptr %117, align 4, !tbaa !10
  %119 = getelementptr inbounds nuw i8, ptr %50, i64 24
  %120 = load float, ptr %119, align 4, !tbaa !10
  %121 = getelementptr inbounds nuw i8, ptr %50, i64 20
  %122 = load float, ptr %121, align 4, !tbaa !10
  %123 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %124 = load float, ptr %123, align 4, !tbaa !10
  %125 = getelementptr inbounds nuw i8, ptr %50, i64 12
  %126 = load float, ptr %125, align 4, !tbaa !10
  %127 = getelementptr inbounds nuw i8, ptr %50, i64 8
  %128 = load float, ptr %127, align 4, !tbaa !10
  %129 = getelementptr inbounds nuw i8, ptr %50, i64 4
  %130 = load float, ptr %129, align 4, !tbaa !10
  %131 = insertelement <4 x float> poison, float %130, i64 0
  %132 = shufflevector <4 x float> %131, <4 x float> poison, <4 x i32> zeroinitializer
  %133 = insertelement <4 x float> poison, float %128, i64 0
  %134 = shufflevector <4 x float> %133, <4 x float> poison, <4 x i32> zeroinitializer
  %135 = insertelement <4 x float> poison, float %126, i64 0
  %136 = shufflevector <4 x float> %135, <4 x float> poison, <4 x i32> zeroinitializer
  %137 = insertelement <4 x float> poison, float %124, i64 0
  %138 = shufflevector <4 x float> %137, <4 x float> poison, <4 x i32> zeroinitializer
  %139 = insertelement <4 x float> poison, float %122, i64 0
  %140 = shufflevector <4 x float> %139, <4 x float> poison, <4 x i32> zeroinitializer
  %141 = insertelement <4 x float> poison, float %120, i64 0
  %142 = shufflevector <4 x float> %141, <4 x float> poison, <4 x i32> zeroinitializer
  %143 = insertelement <4 x float> poison, float %118, i64 0
  %144 = shufflevector <4 x float> %143, <4 x float> poison, <4 x i32> zeroinitializer
  %145 = insertelement <4 x float> poison, float %116, i64 0
  %146 = shufflevector <4 x float> %145, <4 x float> poison, <4 x i32> zeroinitializer
  %147 = insertelement <4 x float> poison, float %114, i64 0
  %148 = shufflevector <4 x float> %147, <4 x float> poison, <4 x i32> zeroinitializer
  %149 = insertelement <4 x float> poison, float %112, i64 0
  %150 = shufflevector <4 x float> %149, <4 x float> poison, <4 x i32> zeroinitializer
  %151 = insertelement <4 x float> poison, float %110, i64 0
  %152 = shufflevector <4 x float> %151, <4 x float> poison, <4 x i32> zeroinitializer
  %153 = insertelement <4 x float> poison, float %108, i64 0
  %154 = shufflevector <4 x float> %153, <4 x float> poison, <4 x i32> zeroinitializer
  %155 = insertelement <4 x float> poison, float %106, i64 0
  %156 = shufflevector <4 x float> %155, <4 x float> poison, <4 x i32> zeroinitializer
  %157 = insertelement <4 x float> poison, float %104, i64 0
  %158 = shufflevector <4 x float> %157, <4 x float> poison, <4 x i32> zeroinitializer
  %159 = insertelement <4 x float> poison, float %102, i64 0
  %160 = shufflevector <4 x float> %159, <4 x float> poison, <4 x i32> zeroinitializer
  %161 = insertelement <4 x float> poison, float %100, i64 0
  %162 = shufflevector <4 x float> %161, <4 x float> poison, <4 x i32> zeroinitializer
  %163 = insertelement <4 x float> poison, float %98, i64 0
  %164 = shufflevector <4 x float> %163, <4 x float> poison, <4 x i32> zeroinitializer
  %165 = insertelement <4 x float> poison, float %96, i64 0
  %166 = shufflevector <4 x float> %165, <4 x float> poison, <4 x i32> zeroinitializer
  %167 = insertelement <4 x float> poison, float %94, i64 0
  %168 = shufflevector <4 x float> %167, <4 x float> poison, <4 x i32> zeroinitializer
  %169 = insertelement <4 x float> poison, float %92, i64 0
  %170 = shufflevector <4 x float> %169, <4 x float> poison, <4 x i32> zeroinitializer
  %171 = insertelement <4 x float> poison, float %90, i64 0
  %172 = shufflevector <4 x float> %171, <4 x float> poison, <4 x i32> zeroinitializer
  %173 = insertelement <4 x float> poison, float %88, i64 0
  %174 = shufflevector <4 x float> %173, <4 x float> poison, <4 x i32> zeroinitializer
  %175 = insertelement <4 x float> poison, float %86, i64 0
  %176 = shufflevector <4 x float> %175, <4 x float> poison, <4 x i32> zeroinitializer
  %177 = insertelement <4 x float> poison, float %84, i64 0
  %178 = shufflevector <4 x float> %177, <4 x float> poison, <4 x i32> zeroinitializer
  %179 = insertelement <4 x float> poison, float %82, i64 0
  %180 = shufflevector <4 x float> %179, <4 x float> poison, <4 x i32> zeroinitializer
  %181 = insertelement <4 x float> poison, float %80, i64 0
  %182 = shufflevector <4 x float> %181, <4 x float> poison, <4 x i32> zeroinitializer
  %183 = insertelement <4 x float> poison, float %78, i64 0
  %184 = shufflevector <4 x float> %183, <4 x float> poison, <4 x i32> zeroinitializer
  %185 = insertelement <4 x float> poison, float %76, i64 0
  %186 = shufflevector <4 x float> %185, <4 x float> poison, <4 x i32> zeroinitializer
  %187 = insertelement <4 x float> poison, float %74, i64 0
  %188 = shufflevector <4 x float> %187, <4 x float> poison, <4 x i32> zeroinitializer
  %189 = insertelement <4 x float> poison, float %72, i64 0
  %190 = shufflevector <4 x float> %189, <4 x float> poison, <4 x i32> zeroinitializer
  %191 = insertelement <4 x float> poison, float %70, i64 0
  %192 = shufflevector <4 x float> %191, <4 x float> poison, <4 x i32> zeroinitializer
  %193 = insertelement <4 x float> poison, float %68, i64 0
  %194 = shufflevector <4 x float> %193, <4 x float> poison, <4 x i32> zeroinitializer
  %195 = insertelement <4 x float> poison, float %66, i64 0
  %196 = shufflevector <4 x float> %195, <4 x float> poison, <4 x i32> zeroinitializer
  %197 = insertelement <4 x float> poison, float %64, i64 0
  %198 = shufflevector <4 x float> %197, <4 x float> poison, <4 x i32> zeroinitializer
  %199 = insertelement <4 x float> poison, float %62, i64 0
  %200 = shufflevector <4 x float> %199, <4 x float> poison, <4 x i32> zeroinitializer
  %201 = insertelement <4 x float> poison, float %60, i64 0
  %202 = shufflevector <4 x float> %201, <4 x float> poison, <4 x i32> zeroinitializer
  %203 = insertelement <4 x float> poison, float %58, i64 0
  %204 = shufflevector <4 x float> %203, <4 x float> poison, <4 x i32> zeroinitializer
  %205 = insertelement <4 x float> poison, float %56, i64 0
  %206 = shufflevector <4 x float> %205, <4 x float> poison, <4 x i32> zeroinitializer
  %207 = insertelement <4 x float> poison, float %54, i64 0
  %208 = shufflevector <4 x float> %207, <4 x float> poison, <4 x i32> zeroinitializer
  %209 = insertelement <4 x float> poison, float %52, i64 0
  %210 = shufflevector <4 x float> %209, <4 x float> poison, <4 x i32> zeroinitializer
  br label %211

211:                                              ; preds = %211, %47
  %212 = phi i64 [ 0, %47 ], [ %336, %211 ]
  %213 = or disjoint i64 %212, 1
  %214 = getelementptr inbounds nuw float, ptr %49, i64 %213
  %215 = getelementptr float, ptr @rmb, i64 %213
  %216 = getelementptr i8, ptr %215, i64 164
  %217 = load <4 x float>, ptr %216, align 4, !tbaa !10
  %218 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %132, <4 x float> %217, <4 x float> zeroinitializer)
  %219 = getelementptr i8, ptr %215, i64 328
  %220 = load <4 x float>, ptr %219, align 4, !tbaa !10
  %221 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %134, <4 x float> %220, <4 x float> %218)
  %222 = getelementptr i8, ptr %215, i64 492
  %223 = load <4 x float>, ptr %222, align 4, !tbaa !10
  %224 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %136, <4 x float> %223, <4 x float> %221)
  %225 = getelementptr i8, ptr %215, i64 656
  %226 = load <4 x float>, ptr %225, align 4, !tbaa !10
  %227 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %138, <4 x float> %226, <4 x float> %224)
  %228 = getelementptr i8, ptr %215, i64 820
  %229 = load <4 x float>, ptr %228, align 4, !tbaa !10
  %230 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %140, <4 x float> %229, <4 x float> %227)
  %231 = getelementptr i8, ptr %215, i64 984
  %232 = load <4 x float>, ptr %231, align 4, !tbaa !10
  %233 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %142, <4 x float> %232, <4 x float> %230)
  %234 = getelementptr i8, ptr %215, i64 1148
  %235 = load <4 x float>, ptr %234, align 4, !tbaa !10
  %236 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %144, <4 x float> %235, <4 x float> %233)
  %237 = getelementptr i8, ptr %215, i64 1312
  %238 = load <4 x float>, ptr %237, align 4, !tbaa !10
  %239 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %146, <4 x float> %238, <4 x float> %236)
  %240 = getelementptr i8, ptr %215, i64 1476
  %241 = load <4 x float>, ptr %240, align 4, !tbaa !10
  %242 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %148, <4 x float> %241, <4 x float> %239)
  %243 = getelementptr i8, ptr %215, i64 1640
  %244 = load <4 x float>, ptr %243, align 4, !tbaa !10
  %245 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %150, <4 x float> %244, <4 x float> %242)
  %246 = getelementptr i8, ptr %215, i64 1804
  %247 = load <4 x float>, ptr %246, align 4, !tbaa !10
  %248 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %152, <4 x float> %247, <4 x float> %245)
  %249 = getelementptr i8, ptr %215, i64 1968
  %250 = load <4 x float>, ptr %249, align 4, !tbaa !10
  %251 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %154, <4 x float> %250, <4 x float> %248)
  %252 = getelementptr i8, ptr %215, i64 2132
  %253 = load <4 x float>, ptr %252, align 4, !tbaa !10
  %254 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %156, <4 x float> %253, <4 x float> %251)
  %255 = getelementptr i8, ptr %215, i64 2296
  %256 = load <4 x float>, ptr %255, align 4, !tbaa !10
  %257 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %158, <4 x float> %256, <4 x float> %254)
  %258 = getelementptr i8, ptr %215, i64 2460
  %259 = load <4 x float>, ptr %258, align 4, !tbaa !10
  %260 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %160, <4 x float> %259, <4 x float> %257)
  %261 = getelementptr i8, ptr %215, i64 2624
  %262 = load <4 x float>, ptr %261, align 4, !tbaa !10
  %263 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %162, <4 x float> %262, <4 x float> %260)
  %264 = getelementptr i8, ptr %215, i64 2788
  %265 = load <4 x float>, ptr %264, align 4, !tbaa !10
  %266 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %164, <4 x float> %265, <4 x float> %263)
  %267 = getelementptr i8, ptr %215, i64 2952
  %268 = load <4 x float>, ptr %267, align 4, !tbaa !10
  %269 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %166, <4 x float> %268, <4 x float> %266)
  %270 = getelementptr i8, ptr %215, i64 3116
  %271 = load <4 x float>, ptr %270, align 4, !tbaa !10
  %272 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %168, <4 x float> %271, <4 x float> %269)
  %273 = getelementptr i8, ptr %215, i64 3280
  %274 = load <4 x float>, ptr %273, align 4, !tbaa !10
  %275 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %170, <4 x float> %274, <4 x float> %272)
  %276 = getelementptr i8, ptr %215, i64 3444
  %277 = load <4 x float>, ptr %276, align 4, !tbaa !10
  %278 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %172, <4 x float> %277, <4 x float> %275)
  %279 = getelementptr i8, ptr %215, i64 3608
  %280 = load <4 x float>, ptr %279, align 4, !tbaa !10
  %281 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %174, <4 x float> %280, <4 x float> %278)
  %282 = getelementptr i8, ptr %215, i64 3772
  %283 = load <4 x float>, ptr %282, align 4, !tbaa !10
  %284 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %176, <4 x float> %283, <4 x float> %281)
  %285 = getelementptr i8, ptr %215, i64 3936
  %286 = load <4 x float>, ptr %285, align 4, !tbaa !10
  %287 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %178, <4 x float> %286, <4 x float> %284)
  %288 = getelementptr i8, ptr %215, i64 4100
  %289 = load <4 x float>, ptr %288, align 4, !tbaa !10
  %290 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %180, <4 x float> %289, <4 x float> %287)
  %291 = getelementptr i8, ptr %215, i64 4264
  %292 = load <4 x float>, ptr %291, align 4, !tbaa !10
  %293 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %182, <4 x float> %292, <4 x float> %290)
  %294 = getelementptr i8, ptr %215, i64 4428
  %295 = load <4 x float>, ptr %294, align 4, !tbaa !10
  %296 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %184, <4 x float> %295, <4 x float> %293)
  %297 = getelementptr i8, ptr %215, i64 4592
  %298 = load <4 x float>, ptr %297, align 4, !tbaa !10
  %299 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %186, <4 x float> %298, <4 x float> %296)
  %300 = getelementptr i8, ptr %215, i64 4756
  %301 = load <4 x float>, ptr %300, align 4, !tbaa !10
  %302 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %188, <4 x float> %301, <4 x float> %299)
  %303 = getelementptr i8, ptr %215, i64 4920
  %304 = load <4 x float>, ptr %303, align 4, !tbaa !10
  %305 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %190, <4 x float> %304, <4 x float> %302)
  %306 = getelementptr i8, ptr %215, i64 5084
  %307 = load <4 x float>, ptr %306, align 4, !tbaa !10
  %308 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %192, <4 x float> %307, <4 x float> %305)
  %309 = getelementptr i8, ptr %215, i64 5248
  %310 = load <4 x float>, ptr %309, align 4, !tbaa !10
  %311 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %194, <4 x float> %310, <4 x float> %308)
  %312 = getelementptr i8, ptr %215, i64 5412
  %313 = load <4 x float>, ptr %312, align 4, !tbaa !10
  %314 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %196, <4 x float> %313, <4 x float> %311)
  %315 = getelementptr i8, ptr %215, i64 5576
  %316 = load <4 x float>, ptr %315, align 4, !tbaa !10
  %317 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %198, <4 x float> %316, <4 x float> %314)
  %318 = getelementptr i8, ptr %215, i64 5740
  %319 = load <4 x float>, ptr %318, align 4, !tbaa !10
  %320 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %200, <4 x float> %319, <4 x float> %317)
  %321 = getelementptr i8, ptr %215, i64 5904
  %322 = load <4 x float>, ptr %321, align 4, !tbaa !10
  %323 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %202, <4 x float> %322, <4 x float> %320)
  %324 = getelementptr i8, ptr %215, i64 6068
  %325 = load <4 x float>, ptr %324, align 4, !tbaa !10
  %326 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %204, <4 x float> %325, <4 x float> %323)
  %327 = getelementptr i8, ptr %215, i64 6232
  %328 = load <4 x float>, ptr %327, align 4, !tbaa !10
  %329 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %206, <4 x float> %328, <4 x float> %326)
  %330 = getelementptr i8, ptr %215, i64 6396
  %331 = load <4 x float>, ptr %330, align 4, !tbaa !10
  %332 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %208, <4 x float> %331, <4 x float> %329)
  %333 = getelementptr i8, ptr %215, i64 6560
  %334 = load <4 x float>, ptr %333, align 4, !tbaa !10
  %335 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %210, <4 x float> %334, <4 x float> %332)
  store <4 x float> %335, ptr %214, align 4, !tbaa !10
  %336 = add nuw i64 %212, 4
  %337 = icmp eq i64 %336, 40
  br i1 %337, label %338, label %211, !llvm.loop !15

338:                                              ; preds = %211
  %339 = add nuw nsw i64 %48, 1
  %340 = icmp eq i64 %339, 41
  br i1 %340, label %341, label %47, !llvm.loop !18

341:                                              ; preds = %338
  %342 = icmp slt i32 %0, 40
  br i1 %342, label %343, label %351

343:                                              ; preds = %341
  %344 = add nsw i32 %0, 1
  %345 = sext i32 %344 to i64
  %346 = getelementptr inbounds [41 x float], ptr @rmr, i64 %345
  %347 = getelementptr inbounds float, ptr %346, i64 %345
  %348 = load float, ptr %347, align 4, !tbaa !10
  %349 = fpext float %348 to double
  %350 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %349)
  br label %351

351:                                              ; preds = %343, %341
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #6

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i32 [ 0, %0 ], [ %3, %1 ]
  tail call void @Mm(i32 noundef %2)
  %3 = add nuw nsw i32 %2, 1
  %4 = icmp eq i32 %3, 5000
  br i1 %4, label %5, label %1, !llvm.loop !19

5:                                                ; preds = %1
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #7

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree norecurse nosync nounwind memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = distinct !{!15, !13, !16, !17}
!16 = !{!"llvm.loop.isvectorized", i32 1}
!17 = !{!"llvm.loop.unroll.runtime.disable"}
!18 = distinct !{!18, !13}
!19 = distinct !{!19, !13}
