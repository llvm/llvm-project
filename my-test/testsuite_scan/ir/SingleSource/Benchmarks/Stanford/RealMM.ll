; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/RealMM.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/RealMM.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.element = type { i32, i32 }
%struct.complex = type { double, double }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@rma = dso_local local_unnamed_addr global [41 x [41 x double]] zeroinitializer, align 8
@rmb = dso_local local_unnamed_addr global [41 x [41 x double]] zeroinitializer, align 8
@rmr = dso_local local_unnamed_addr global [41 x [41 x double]] zeroinitializer, align 8
@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@value = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@fixed = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@floated = dso_local local_unnamed_addr global double 0.000000e+00, align 8
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
@z = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 8
@w = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 8
@e = dso_local local_unnamed_addr global [130 x %struct.complex] zeroinitializer, align 8
@zr = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@zi = dso_local local_unnamed_addr global double 0.000000e+00, align 8

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
  %7 = getelementptr inbounds nuw [41 x double], ptr %0, i64 %5
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
  %18 = sitofp i32 %17 to double
  %19 = fdiv double %18, 3.000000e+00
  %20 = getelementptr inbounds nuw double, ptr %7, i64 %9
  store double %19, ptr %20, align 8, !tbaa !10
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
define dso_local void @rInnerproduct(ptr noundef writeonly captures(none) initializes((0, 8)) %0, ptr noundef readonly captures(none) %1, ptr noundef readonly captures(none) %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #3 {
  store double 0.000000e+00, ptr %0, align 8, !tbaa !10
  %6 = sext i32 %3 to i64
  %7 = getelementptr inbounds [41 x double], ptr %1, i64 %6
  %8 = sext i32 %4 to i64
  %9 = getelementptr double, ptr %2, i64 %8
  %10 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %11 = load double, ptr %10, align 8, !tbaa !10
  %12 = getelementptr i8, ptr %9, i64 328
  %13 = load double, ptr %12, align 8, !tbaa !10
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double 0.000000e+00)
  store double %14, ptr %0, align 8, !tbaa !10
  %15 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %16 = load double, ptr %15, align 8, !tbaa !10
  %17 = getelementptr i8, ptr %9, i64 656
  %18 = load double, ptr %17, align 8, !tbaa !10
  %19 = tail call double @llvm.fmuladd.f64(double %16, double %18, double %14)
  store double %19, ptr %0, align 8, !tbaa !10
  %20 = getelementptr inbounds nuw i8, ptr %7, i64 24
  %21 = load double, ptr %20, align 8, !tbaa !10
  %22 = getelementptr i8, ptr %9, i64 984
  %23 = load double, ptr %22, align 8, !tbaa !10
  %24 = tail call double @llvm.fmuladd.f64(double %21, double %23, double %19)
  store double %24, ptr %0, align 8, !tbaa !10
  %25 = getelementptr inbounds nuw i8, ptr %7, i64 32
  %26 = load double, ptr %25, align 8, !tbaa !10
  %27 = getelementptr i8, ptr %9, i64 1312
  %28 = load double, ptr %27, align 8, !tbaa !10
  %29 = tail call double @llvm.fmuladd.f64(double %26, double %28, double %24)
  store double %29, ptr %0, align 8, !tbaa !10
  %30 = getelementptr inbounds nuw i8, ptr %7, i64 40
  %31 = load double, ptr %30, align 8, !tbaa !10
  %32 = getelementptr i8, ptr %9, i64 1640
  %33 = load double, ptr %32, align 8, !tbaa !10
  %34 = tail call double @llvm.fmuladd.f64(double %31, double %33, double %29)
  store double %34, ptr %0, align 8, !tbaa !10
  %35 = getelementptr inbounds nuw i8, ptr %7, i64 48
  %36 = load double, ptr %35, align 8, !tbaa !10
  %37 = getelementptr i8, ptr %9, i64 1968
  %38 = load double, ptr %37, align 8, !tbaa !10
  %39 = tail call double @llvm.fmuladd.f64(double %36, double %38, double %34)
  store double %39, ptr %0, align 8, !tbaa !10
  %40 = getelementptr inbounds nuw i8, ptr %7, i64 56
  %41 = load double, ptr %40, align 8, !tbaa !10
  %42 = getelementptr i8, ptr %9, i64 2296
  %43 = load double, ptr %42, align 8, !tbaa !10
  %44 = tail call double @llvm.fmuladd.f64(double %41, double %43, double %39)
  store double %44, ptr %0, align 8, !tbaa !10
  %45 = getelementptr inbounds nuw i8, ptr %7, i64 64
  %46 = load double, ptr %45, align 8, !tbaa !10
  %47 = getelementptr i8, ptr %9, i64 2624
  %48 = load double, ptr %47, align 8, !tbaa !10
  %49 = tail call double @llvm.fmuladd.f64(double %46, double %48, double %44)
  store double %49, ptr %0, align 8, !tbaa !10
  %50 = getelementptr inbounds nuw i8, ptr %7, i64 72
  %51 = load double, ptr %50, align 8, !tbaa !10
  %52 = getelementptr i8, ptr %9, i64 2952
  %53 = load double, ptr %52, align 8, !tbaa !10
  %54 = tail call double @llvm.fmuladd.f64(double %51, double %53, double %49)
  store double %54, ptr %0, align 8, !tbaa !10
  %55 = getelementptr inbounds nuw i8, ptr %7, i64 80
  %56 = load double, ptr %55, align 8, !tbaa !10
  %57 = getelementptr i8, ptr %9, i64 3280
  %58 = load double, ptr %57, align 8, !tbaa !10
  %59 = tail call double @llvm.fmuladd.f64(double %56, double %58, double %54)
  store double %59, ptr %0, align 8, !tbaa !10
  %60 = getelementptr inbounds nuw i8, ptr %7, i64 88
  %61 = load double, ptr %60, align 8, !tbaa !10
  %62 = getelementptr i8, ptr %9, i64 3608
  %63 = load double, ptr %62, align 8, !tbaa !10
  %64 = tail call double @llvm.fmuladd.f64(double %61, double %63, double %59)
  store double %64, ptr %0, align 8, !tbaa !10
  %65 = getelementptr inbounds nuw i8, ptr %7, i64 96
  %66 = load double, ptr %65, align 8, !tbaa !10
  %67 = getelementptr i8, ptr %9, i64 3936
  %68 = load double, ptr %67, align 8, !tbaa !10
  %69 = tail call double @llvm.fmuladd.f64(double %66, double %68, double %64)
  store double %69, ptr %0, align 8, !tbaa !10
  %70 = getelementptr inbounds nuw i8, ptr %7, i64 104
  %71 = load double, ptr %70, align 8, !tbaa !10
  %72 = getelementptr i8, ptr %9, i64 4264
  %73 = load double, ptr %72, align 8, !tbaa !10
  %74 = tail call double @llvm.fmuladd.f64(double %71, double %73, double %69)
  store double %74, ptr %0, align 8, !tbaa !10
  %75 = getelementptr inbounds nuw i8, ptr %7, i64 112
  %76 = load double, ptr %75, align 8, !tbaa !10
  %77 = getelementptr i8, ptr %9, i64 4592
  %78 = load double, ptr %77, align 8, !tbaa !10
  %79 = tail call double @llvm.fmuladd.f64(double %76, double %78, double %74)
  store double %79, ptr %0, align 8, !tbaa !10
  %80 = getelementptr inbounds nuw i8, ptr %7, i64 120
  %81 = load double, ptr %80, align 8, !tbaa !10
  %82 = getelementptr i8, ptr %9, i64 4920
  %83 = load double, ptr %82, align 8, !tbaa !10
  %84 = tail call double @llvm.fmuladd.f64(double %81, double %83, double %79)
  store double %84, ptr %0, align 8, !tbaa !10
  %85 = getelementptr inbounds nuw i8, ptr %7, i64 128
  %86 = load double, ptr %85, align 8, !tbaa !10
  %87 = getelementptr i8, ptr %9, i64 5248
  %88 = load double, ptr %87, align 8, !tbaa !10
  %89 = tail call double @llvm.fmuladd.f64(double %86, double %88, double %84)
  store double %89, ptr %0, align 8, !tbaa !10
  %90 = getelementptr inbounds nuw i8, ptr %7, i64 136
  %91 = load double, ptr %90, align 8, !tbaa !10
  %92 = getelementptr i8, ptr %9, i64 5576
  %93 = load double, ptr %92, align 8, !tbaa !10
  %94 = tail call double @llvm.fmuladd.f64(double %91, double %93, double %89)
  store double %94, ptr %0, align 8, !tbaa !10
  %95 = getelementptr inbounds nuw i8, ptr %7, i64 144
  %96 = load double, ptr %95, align 8, !tbaa !10
  %97 = getelementptr i8, ptr %9, i64 5904
  %98 = load double, ptr %97, align 8, !tbaa !10
  %99 = tail call double @llvm.fmuladd.f64(double %96, double %98, double %94)
  store double %99, ptr %0, align 8, !tbaa !10
  %100 = getelementptr inbounds nuw i8, ptr %7, i64 152
  %101 = load double, ptr %100, align 8, !tbaa !10
  %102 = getelementptr i8, ptr %9, i64 6232
  %103 = load double, ptr %102, align 8, !tbaa !10
  %104 = tail call double @llvm.fmuladd.f64(double %101, double %103, double %99)
  store double %104, ptr %0, align 8, !tbaa !10
  %105 = getelementptr inbounds nuw i8, ptr %7, i64 160
  %106 = load double, ptr %105, align 8, !tbaa !10
  %107 = getelementptr i8, ptr %9, i64 6560
  %108 = load double, ptr %107, align 8, !tbaa !10
  %109 = tail call double @llvm.fmuladd.f64(double %106, double %108, double %104)
  store double %109, ptr %0, align 8, !tbaa !10
  %110 = getelementptr inbounds nuw i8, ptr %7, i64 168
  %111 = load double, ptr %110, align 8, !tbaa !10
  %112 = getelementptr i8, ptr %9, i64 6888
  %113 = load double, ptr %112, align 8, !tbaa !10
  %114 = tail call double @llvm.fmuladd.f64(double %111, double %113, double %109)
  store double %114, ptr %0, align 8, !tbaa !10
  %115 = getelementptr inbounds nuw i8, ptr %7, i64 176
  %116 = load double, ptr %115, align 8, !tbaa !10
  %117 = getelementptr i8, ptr %9, i64 7216
  %118 = load double, ptr %117, align 8, !tbaa !10
  %119 = tail call double @llvm.fmuladd.f64(double %116, double %118, double %114)
  store double %119, ptr %0, align 8, !tbaa !10
  %120 = getelementptr inbounds nuw i8, ptr %7, i64 184
  %121 = load double, ptr %120, align 8, !tbaa !10
  %122 = getelementptr i8, ptr %9, i64 7544
  %123 = load double, ptr %122, align 8, !tbaa !10
  %124 = tail call double @llvm.fmuladd.f64(double %121, double %123, double %119)
  store double %124, ptr %0, align 8, !tbaa !10
  %125 = getelementptr inbounds nuw i8, ptr %7, i64 192
  %126 = load double, ptr %125, align 8, !tbaa !10
  %127 = getelementptr i8, ptr %9, i64 7872
  %128 = load double, ptr %127, align 8, !tbaa !10
  %129 = tail call double @llvm.fmuladd.f64(double %126, double %128, double %124)
  store double %129, ptr %0, align 8, !tbaa !10
  %130 = getelementptr inbounds nuw i8, ptr %7, i64 200
  %131 = load double, ptr %130, align 8, !tbaa !10
  %132 = getelementptr i8, ptr %9, i64 8200
  %133 = load double, ptr %132, align 8, !tbaa !10
  %134 = tail call double @llvm.fmuladd.f64(double %131, double %133, double %129)
  store double %134, ptr %0, align 8, !tbaa !10
  %135 = getelementptr inbounds nuw i8, ptr %7, i64 208
  %136 = load double, ptr %135, align 8, !tbaa !10
  %137 = getelementptr i8, ptr %9, i64 8528
  %138 = load double, ptr %137, align 8, !tbaa !10
  %139 = tail call double @llvm.fmuladd.f64(double %136, double %138, double %134)
  store double %139, ptr %0, align 8, !tbaa !10
  %140 = getelementptr inbounds nuw i8, ptr %7, i64 216
  %141 = load double, ptr %140, align 8, !tbaa !10
  %142 = getelementptr i8, ptr %9, i64 8856
  %143 = load double, ptr %142, align 8, !tbaa !10
  %144 = tail call double @llvm.fmuladd.f64(double %141, double %143, double %139)
  store double %144, ptr %0, align 8, !tbaa !10
  %145 = getelementptr inbounds nuw i8, ptr %7, i64 224
  %146 = load double, ptr %145, align 8, !tbaa !10
  %147 = getelementptr i8, ptr %9, i64 9184
  %148 = load double, ptr %147, align 8, !tbaa !10
  %149 = tail call double @llvm.fmuladd.f64(double %146, double %148, double %144)
  store double %149, ptr %0, align 8, !tbaa !10
  %150 = getelementptr inbounds nuw i8, ptr %7, i64 232
  %151 = load double, ptr %150, align 8, !tbaa !10
  %152 = getelementptr i8, ptr %9, i64 9512
  %153 = load double, ptr %152, align 8, !tbaa !10
  %154 = tail call double @llvm.fmuladd.f64(double %151, double %153, double %149)
  store double %154, ptr %0, align 8, !tbaa !10
  %155 = getelementptr inbounds nuw i8, ptr %7, i64 240
  %156 = load double, ptr %155, align 8, !tbaa !10
  %157 = getelementptr i8, ptr %9, i64 9840
  %158 = load double, ptr %157, align 8, !tbaa !10
  %159 = tail call double @llvm.fmuladd.f64(double %156, double %158, double %154)
  store double %159, ptr %0, align 8, !tbaa !10
  %160 = getelementptr inbounds nuw i8, ptr %7, i64 248
  %161 = load double, ptr %160, align 8, !tbaa !10
  %162 = getelementptr i8, ptr %9, i64 10168
  %163 = load double, ptr %162, align 8, !tbaa !10
  %164 = tail call double @llvm.fmuladd.f64(double %161, double %163, double %159)
  store double %164, ptr %0, align 8, !tbaa !10
  %165 = getelementptr inbounds nuw i8, ptr %7, i64 256
  %166 = load double, ptr %165, align 8, !tbaa !10
  %167 = getelementptr i8, ptr %9, i64 10496
  %168 = load double, ptr %167, align 8, !tbaa !10
  %169 = tail call double @llvm.fmuladd.f64(double %166, double %168, double %164)
  store double %169, ptr %0, align 8, !tbaa !10
  %170 = getelementptr inbounds nuw i8, ptr %7, i64 264
  %171 = load double, ptr %170, align 8, !tbaa !10
  %172 = getelementptr i8, ptr %9, i64 10824
  %173 = load double, ptr %172, align 8, !tbaa !10
  %174 = tail call double @llvm.fmuladd.f64(double %171, double %173, double %169)
  store double %174, ptr %0, align 8, !tbaa !10
  %175 = getelementptr inbounds nuw i8, ptr %7, i64 272
  %176 = load double, ptr %175, align 8, !tbaa !10
  %177 = getelementptr i8, ptr %9, i64 11152
  %178 = load double, ptr %177, align 8, !tbaa !10
  %179 = tail call double @llvm.fmuladd.f64(double %176, double %178, double %174)
  store double %179, ptr %0, align 8, !tbaa !10
  %180 = getelementptr inbounds nuw i8, ptr %7, i64 280
  %181 = load double, ptr %180, align 8, !tbaa !10
  %182 = getelementptr i8, ptr %9, i64 11480
  %183 = load double, ptr %182, align 8, !tbaa !10
  %184 = tail call double @llvm.fmuladd.f64(double %181, double %183, double %179)
  store double %184, ptr %0, align 8, !tbaa !10
  %185 = getelementptr inbounds nuw i8, ptr %7, i64 288
  %186 = load double, ptr %185, align 8, !tbaa !10
  %187 = getelementptr i8, ptr %9, i64 11808
  %188 = load double, ptr %187, align 8, !tbaa !10
  %189 = tail call double @llvm.fmuladd.f64(double %186, double %188, double %184)
  store double %189, ptr %0, align 8, !tbaa !10
  %190 = getelementptr inbounds nuw i8, ptr %7, i64 296
  %191 = load double, ptr %190, align 8, !tbaa !10
  %192 = getelementptr i8, ptr %9, i64 12136
  %193 = load double, ptr %192, align 8, !tbaa !10
  %194 = tail call double @llvm.fmuladd.f64(double %191, double %193, double %189)
  store double %194, ptr %0, align 8, !tbaa !10
  %195 = getelementptr inbounds nuw i8, ptr %7, i64 304
  %196 = load double, ptr %195, align 8, !tbaa !10
  %197 = getelementptr i8, ptr %9, i64 12464
  %198 = load double, ptr %197, align 8, !tbaa !10
  %199 = tail call double @llvm.fmuladd.f64(double %196, double %198, double %194)
  store double %199, ptr %0, align 8, !tbaa !10
  %200 = getelementptr inbounds nuw i8, ptr %7, i64 312
  %201 = load double, ptr %200, align 8, !tbaa !10
  %202 = getelementptr i8, ptr %9, i64 12792
  %203 = load double, ptr %202, align 8, !tbaa !10
  %204 = tail call double @llvm.fmuladd.f64(double %201, double %203, double %199)
  store double %204, ptr %0, align 8, !tbaa !10
  %205 = getelementptr inbounds nuw i8, ptr %7, i64 320
  %206 = load double, ptr %205, align 8, !tbaa !10
  %207 = getelementptr i8, ptr %9, i64 13120
  %208 = load double, ptr %207, align 8, !tbaa !10
  %209 = tail call double @llvm.fmuladd.f64(double %206, double %208, double %204)
  store double %209, ptr %0, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #4

; Function Attrs: nofree nounwind uwtable
define dso_local void @Mm(i32 noundef %0) local_unnamed_addr #5 {
  br label %2

2:                                                ; preds = %21, %1
  %3 = phi i64 [ 1, %1 ], [ %22, %21 ]
  %4 = phi i64 [ 74755, %1 ], [ %11, %21 ]
  %5 = getelementptr inbounds nuw [41 x double], ptr @rma, i64 %3
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
  %16 = sitofp i32 %15 to double
  %17 = fdiv double %16, 3.000000e+00
  %18 = getelementptr inbounds nuw double, ptr %5, i64 %7
  store double %17, ptr %18, align 8, !tbaa !10
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
  %27 = getelementptr inbounds nuw [41 x double], ptr @rmb, i64 %25
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
  %38 = sitofp i32 %37 to double
  %39 = fdiv double %38, 3.000000e+00
  %40 = getelementptr inbounds nuw double, ptr %27, i64 %29
  store double %39, ptr %40, align 8, !tbaa !10
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
  %49 = getelementptr inbounds nuw [41 x double], ptr @rmr, i64 %48
  %50 = getelementptr inbounds nuw [41 x double], ptr @rma, i64 %48
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 320
  %52 = load double, ptr %51, align 8, !tbaa !10
  %53 = getelementptr inbounds nuw i8, ptr %50, i64 312
  %54 = load double, ptr %53, align 8, !tbaa !10
  %55 = getelementptr inbounds nuw i8, ptr %50, i64 304
  %56 = load double, ptr %55, align 8, !tbaa !10
  %57 = getelementptr inbounds nuw i8, ptr %50, i64 296
  %58 = load double, ptr %57, align 8, !tbaa !10
  %59 = getelementptr inbounds nuw i8, ptr %50, i64 288
  %60 = load double, ptr %59, align 8, !tbaa !10
  %61 = getelementptr inbounds nuw i8, ptr %50, i64 280
  %62 = load double, ptr %61, align 8, !tbaa !10
  %63 = getelementptr inbounds nuw i8, ptr %50, i64 272
  %64 = load double, ptr %63, align 8, !tbaa !10
  %65 = getelementptr inbounds nuw i8, ptr %50, i64 264
  %66 = load double, ptr %65, align 8, !tbaa !10
  %67 = getelementptr inbounds nuw i8, ptr %50, i64 256
  %68 = load double, ptr %67, align 8, !tbaa !10
  %69 = getelementptr inbounds nuw i8, ptr %50, i64 248
  %70 = load double, ptr %69, align 8, !tbaa !10
  %71 = getelementptr inbounds nuw i8, ptr %50, i64 240
  %72 = load double, ptr %71, align 8, !tbaa !10
  %73 = getelementptr inbounds nuw i8, ptr %50, i64 232
  %74 = load double, ptr %73, align 8, !tbaa !10
  %75 = getelementptr inbounds nuw i8, ptr %50, i64 224
  %76 = load double, ptr %75, align 8, !tbaa !10
  %77 = getelementptr inbounds nuw i8, ptr %50, i64 216
  %78 = load double, ptr %77, align 8, !tbaa !10
  %79 = getelementptr inbounds nuw i8, ptr %50, i64 208
  %80 = load double, ptr %79, align 8, !tbaa !10
  %81 = getelementptr inbounds nuw i8, ptr %50, i64 200
  %82 = load double, ptr %81, align 8, !tbaa !10
  %83 = getelementptr inbounds nuw i8, ptr %50, i64 192
  %84 = load double, ptr %83, align 8, !tbaa !10
  %85 = getelementptr inbounds nuw i8, ptr %50, i64 184
  %86 = load double, ptr %85, align 8, !tbaa !10
  %87 = getelementptr inbounds nuw i8, ptr %50, i64 176
  %88 = load double, ptr %87, align 8, !tbaa !10
  %89 = getelementptr inbounds nuw i8, ptr %50, i64 168
  %90 = load double, ptr %89, align 8, !tbaa !10
  %91 = getelementptr inbounds nuw i8, ptr %50, i64 160
  %92 = load double, ptr %91, align 8, !tbaa !10
  %93 = getelementptr inbounds nuw i8, ptr %50, i64 152
  %94 = load double, ptr %93, align 8, !tbaa !10
  %95 = getelementptr inbounds nuw i8, ptr %50, i64 144
  %96 = load double, ptr %95, align 8, !tbaa !10
  %97 = getelementptr inbounds nuw i8, ptr %50, i64 136
  %98 = load double, ptr %97, align 8, !tbaa !10
  %99 = getelementptr inbounds nuw i8, ptr %50, i64 128
  %100 = load double, ptr %99, align 8, !tbaa !10
  %101 = getelementptr inbounds nuw i8, ptr %50, i64 120
  %102 = load double, ptr %101, align 8, !tbaa !10
  %103 = getelementptr inbounds nuw i8, ptr %50, i64 112
  %104 = load double, ptr %103, align 8, !tbaa !10
  %105 = getelementptr inbounds nuw i8, ptr %50, i64 104
  %106 = load double, ptr %105, align 8, !tbaa !10
  %107 = getelementptr inbounds nuw i8, ptr %50, i64 96
  %108 = load double, ptr %107, align 8, !tbaa !10
  %109 = getelementptr inbounds nuw i8, ptr %50, i64 88
  %110 = load double, ptr %109, align 8, !tbaa !10
  %111 = getelementptr inbounds nuw i8, ptr %50, i64 80
  %112 = load double, ptr %111, align 8, !tbaa !10
  %113 = getelementptr inbounds nuw i8, ptr %50, i64 72
  %114 = load double, ptr %113, align 8, !tbaa !10
  %115 = getelementptr inbounds nuw i8, ptr %50, i64 64
  %116 = load double, ptr %115, align 8, !tbaa !10
  %117 = getelementptr inbounds nuw i8, ptr %50, i64 56
  %118 = load double, ptr %117, align 8, !tbaa !10
  %119 = getelementptr inbounds nuw i8, ptr %50, i64 48
  %120 = load double, ptr %119, align 8, !tbaa !10
  %121 = getelementptr inbounds nuw i8, ptr %50, i64 40
  %122 = load double, ptr %121, align 8, !tbaa !10
  %123 = getelementptr inbounds nuw i8, ptr %50, i64 32
  %124 = load double, ptr %123, align 8, !tbaa !10
  %125 = getelementptr inbounds nuw i8, ptr %50, i64 24
  %126 = load double, ptr %125, align 8, !tbaa !10
  %127 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %128 = load double, ptr %127, align 8, !tbaa !10
  %129 = getelementptr inbounds nuw i8, ptr %50, i64 8
  %130 = load double, ptr %129, align 8, !tbaa !10
  %131 = insertelement <2 x double> poison, double %130, i64 0
  %132 = shufflevector <2 x double> %131, <2 x double> poison, <2 x i32> zeroinitializer
  %133 = insertelement <2 x double> poison, double %128, i64 0
  %134 = shufflevector <2 x double> %133, <2 x double> poison, <2 x i32> zeroinitializer
  %135 = insertelement <2 x double> poison, double %126, i64 0
  %136 = shufflevector <2 x double> %135, <2 x double> poison, <2 x i32> zeroinitializer
  %137 = insertelement <2 x double> poison, double %124, i64 0
  %138 = shufflevector <2 x double> %137, <2 x double> poison, <2 x i32> zeroinitializer
  %139 = insertelement <2 x double> poison, double %122, i64 0
  %140 = shufflevector <2 x double> %139, <2 x double> poison, <2 x i32> zeroinitializer
  %141 = insertelement <2 x double> poison, double %120, i64 0
  %142 = shufflevector <2 x double> %141, <2 x double> poison, <2 x i32> zeroinitializer
  %143 = insertelement <2 x double> poison, double %118, i64 0
  %144 = shufflevector <2 x double> %143, <2 x double> poison, <2 x i32> zeroinitializer
  %145 = insertelement <2 x double> poison, double %116, i64 0
  %146 = shufflevector <2 x double> %145, <2 x double> poison, <2 x i32> zeroinitializer
  %147 = insertelement <2 x double> poison, double %114, i64 0
  %148 = shufflevector <2 x double> %147, <2 x double> poison, <2 x i32> zeroinitializer
  %149 = insertelement <2 x double> poison, double %112, i64 0
  %150 = shufflevector <2 x double> %149, <2 x double> poison, <2 x i32> zeroinitializer
  %151 = insertelement <2 x double> poison, double %110, i64 0
  %152 = shufflevector <2 x double> %151, <2 x double> poison, <2 x i32> zeroinitializer
  %153 = insertelement <2 x double> poison, double %108, i64 0
  %154 = shufflevector <2 x double> %153, <2 x double> poison, <2 x i32> zeroinitializer
  %155 = insertelement <2 x double> poison, double %106, i64 0
  %156 = shufflevector <2 x double> %155, <2 x double> poison, <2 x i32> zeroinitializer
  %157 = insertelement <2 x double> poison, double %104, i64 0
  %158 = shufflevector <2 x double> %157, <2 x double> poison, <2 x i32> zeroinitializer
  %159 = insertelement <2 x double> poison, double %102, i64 0
  %160 = shufflevector <2 x double> %159, <2 x double> poison, <2 x i32> zeroinitializer
  %161 = insertelement <2 x double> poison, double %100, i64 0
  %162 = shufflevector <2 x double> %161, <2 x double> poison, <2 x i32> zeroinitializer
  %163 = insertelement <2 x double> poison, double %98, i64 0
  %164 = shufflevector <2 x double> %163, <2 x double> poison, <2 x i32> zeroinitializer
  %165 = insertelement <2 x double> poison, double %96, i64 0
  %166 = shufflevector <2 x double> %165, <2 x double> poison, <2 x i32> zeroinitializer
  %167 = insertelement <2 x double> poison, double %94, i64 0
  %168 = shufflevector <2 x double> %167, <2 x double> poison, <2 x i32> zeroinitializer
  %169 = insertelement <2 x double> poison, double %92, i64 0
  %170 = shufflevector <2 x double> %169, <2 x double> poison, <2 x i32> zeroinitializer
  %171 = insertelement <2 x double> poison, double %90, i64 0
  %172 = shufflevector <2 x double> %171, <2 x double> poison, <2 x i32> zeroinitializer
  %173 = insertelement <2 x double> poison, double %88, i64 0
  %174 = shufflevector <2 x double> %173, <2 x double> poison, <2 x i32> zeroinitializer
  %175 = insertelement <2 x double> poison, double %86, i64 0
  %176 = shufflevector <2 x double> %175, <2 x double> poison, <2 x i32> zeroinitializer
  %177 = insertelement <2 x double> poison, double %84, i64 0
  %178 = shufflevector <2 x double> %177, <2 x double> poison, <2 x i32> zeroinitializer
  %179 = insertelement <2 x double> poison, double %82, i64 0
  %180 = shufflevector <2 x double> %179, <2 x double> poison, <2 x i32> zeroinitializer
  %181 = insertelement <2 x double> poison, double %80, i64 0
  %182 = shufflevector <2 x double> %181, <2 x double> poison, <2 x i32> zeroinitializer
  %183 = insertelement <2 x double> poison, double %78, i64 0
  %184 = shufflevector <2 x double> %183, <2 x double> poison, <2 x i32> zeroinitializer
  %185 = insertelement <2 x double> poison, double %76, i64 0
  %186 = shufflevector <2 x double> %185, <2 x double> poison, <2 x i32> zeroinitializer
  %187 = insertelement <2 x double> poison, double %74, i64 0
  %188 = shufflevector <2 x double> %187, <2 x double> poison, <2 x i32> zeroinitializer
  %189 = insertelement <2 x double> poison, double %72, i64 0
  %190 = shufflevector <2 x double> %189, <2 x double> poison, <2 x i32> zeroinitializer
  %191 = insertelement <2 x double> poison, double %70, i64 0
  %192 = shufflevector <2 x double> %191, <2 x double> poison, <2 x i32> zeroinitializer
  %193 = insertelement <2 x double> poison, double %68, i64 0
  %194 = shufflevector <2 x double> %193, <2 x double> poison, <2 x i32> zeroinitializer
  %195 = insertelement <2 x double> poison, double %66, i64 0
  %196 = shufflevector <2 x double> %195, <2 x double> poison, <2 x i32> zeroinitializer
  %197 = insertelement <2 x double> poison, double %64, i64 0
  %198 = shufflevector <2 x double> %197, <2 x double> poison, <2 x i32> zeroinitializer
  %199 = insertelement <2 x double> poison, double %62, i64 0
  %200 = shufflevector <2 x double> %199, <2 x double> poison, <2 x i32> zeroinitializer
  %201 = insertelement <2 x double> poison, double %60, i64 0
  %202 = shufflevector <2 x double> %201, <2 x double> poison, <2 x i32> zeroinitializer
  %203 = insertelement <2 x double> poison, double %58, i64 0
  %204 = shufflevector <2 x double> %203, <2 x double> poison, <2 x i32> zeroinitializer
  %205 = insertelement <2 x double> poison, double %56, i64 0
  %206 = shufflevector <2 x double> %205, <2 x double> poison, <2 x i32> zeroinitializer
  %207 = insertelement <2 x double> poison, double %54, i64 0
  %208 = shufflevector <2 x double> %207, <2 x double> poison, <2 x i32> zeroinitializer
  %209 = insertelement <2 x double> poison, double %52, i64 0
  %210 = shufflevector <2 x double> %209, <2 x double> poison, <2 x i32> zeroinitializer
  br label %211

211:                                              ; preds = %211, %47
  %212 = phi i64 [ 0, %47 ], [ %336, %211 ]
  %213 = or disjoint i64 %212, 1
  %214 = getelementptr inbounds nuw double, ptr %49, i64 %213
  %215 = getelementptr double, ptr @rmb, i64 %213
  %216 = getelementptr i8, ptr %215, i64 328
  %217 = load <2 x double>, ptr %216, align 8, !tbaa !10
  %218 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %132, <2 x double> %217, <2 x double> zeroinitializer)
  %219 = getelementptr i8, ptr %215, i64 656
  %220 = load <2 x double>, ptr %219, align 8, !tbaa !10
  %221 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %134, <2 x double> %220, <2 x double> %218)
  %222 = getelementptr i8, ptr %215, i64 984
  %223 = load <2 x double>, ptr %222, align 8, !tbaa !10
  %224 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %136, <2 x double> %223, <2 x double> %221)
  %225 = getelementptr i8, ptr %215, i64 1312
  %226 = load <2 x double>, ptr %225, align 8, !tbaa !10
  %227 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %138, <2 x double> %226, <2 x double> %224)
  %228 = getelementptr i8, ptr %215, i64 1640
  %229 = load <2 x double>, ptr %228, align 8, !tbaa !10
  %230 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %140, <2 x double> %229, <2 x double> %227)
  %231 = getelementptr i8, ptr %215, i64 1968
  %232 = load <2 x double>, ptr %231, align 8, !tbaa !10
  %233 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %142, <2 x double> %232, <2 x double> %230)
  %234 = getelementptr i8, ptr %215, i64 2296
  %235 = load <2 x double>, ptr %234, align 8, !tbaa !10
  %236 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %144, <2 x double> %235, <2 x double> %233)
  %237 = getelementptr i8, ptr %215, i64 2624
  %238 = load <2 x double>, ptr %237, align 8, !tbaa !10
  %239 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %146, <2 x double> %238, <2 x double> %236)
  %240 = getelementptr i8, ptr %215, i64 2952
  %241 = load <2 x double>, ptr %240, align 8, !tbaa !10
  %242 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %148, <2 x double> %241, <2 x double> %239)
  %243 = getelementptr i8, ptr %215, i64 3280
  %244 = load <2 x double>, ptr %243, align 8, !tbaa !10
  %245 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %150, <2 x double> %244, <2 x double> %242)
  %246 = getelementptr i8, ptr %215, i64 3608
  %247 = load <2 x double>, ptr %246, align 8, !tbaa !10
  %248 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %152, <2 x double> %247, <2 x double> %245)
  %249 = getelementptr i8, ptr %215, i64 3936
  %250 = load <2 x double>, ptr %249, align 8, !tbaa !10
  %251 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %154, <2 x double> %250, <2 x double> %248)
  %252 = getelementptr i8, ptr %215, i64 4264
  %253 = load <2 x double>, ptr %252, align 8, !tbaa !10
  %254 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %156, <2 x double> %253, <2 x double> %251)
  %255 = getelementptr i8, ptr %215, i64 4592
  %256 = load <2 x double>, ptr %255, align 8, !tbaa !10
  %257 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %158, <2 x double> %256, <2 x double> %254)
  %258 = getelementptr i8, ptr %215, i64 4920
  %259 = load <2 x double>, ptr %258, align 8, !tbaa !10
  %260 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %160, <2 x double> %259, <2 x double> %257)
  %261 = getelementptr i8, ptr %215, i64 5248
  %262 = load <2 x double>, ptr %261, align 8, !tbaa !10
  %263 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %162, <2 x double> %262, <2 x double> %260)
  %264 = getelementptr i8, ptr %215, i64 5576
  %265 = load <2 x double>, ptr %264, align 8, !tbaa !10
  %266 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %164, <2 x double> %265, <2 x double> %263)
  %267 = getelementptr i8, ptr %215, i64 5904
  %268 = load <2 x double>, ptr %267, align 8, !tbaa !10
  %269 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %166, <2 x double> %268, <2 x double> %266)
  %270 = getelementptr i8, ptr %215, i64 6232
  %271 = load <2 x double>, ptr %270, align 8, !tbaa !10
  %272 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %168, <2 x double> %271, <2 x double> %269)
  %273 = getelementptr i8, ptr %215, i64 6560
  %274 = load <2 x double>, ptr %273, align 8, !tbaa !10
  %275 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %170, <2 x double> %274, <2 x double> %272)
  %276 = getelementptr i8, ptr %215, i64 6888
  %277 = load <2 x double>, ptr %276, align 8, !tbaa !10
  %278 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %172, <2 x double> %277, <2 x double> %275)
  %279 = getelementptr i8, ptr %215, i64 7216
  %280 = load <2 x double>, ptr %279, align 8, !tbaa !10
  %281 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %174, <2 x double> %280, <2 x double> %278)
  %282 = getelementptr i8, ptr %215, i64 7544
  %283 = load <2 x double>, ptr %282, align 8, !tbaa !10
  %284 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %176, <2 x double> %283, <2 x double> %281)
  %285 = getelementptr i8, ptr %215, i64 7872
  %286 = load <2 x double>, ptr %285, align 8, !tbaa !10
  %287 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %178, <2 x double> %286, <2 x double> %284)
  %288 = getelementptr i8, ptr %215, i64 8200
  %289 = load <2 x double>, ptr %288, align 8, !tbaa !10
  %290 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %180, <2 x double> %289, <2 x double> %287)
  %291 = getelementptr i8, ptr %215, i64 8528
  %292 = load <2 x double>, ptr %291, align 8, !tbaa !10
  %293 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %182, <2 x double> %292, <2 x double> %290)
  %294 = getelementptr i8, ptr %215, i64 8856
  %295 = load <2 x double>, ptr %294, align 8, !tbaa !10
  %296 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %184, <2 x double> %295, <2 x double> %293)
  %297 = getelementptr i8, ptr %215, i64 9184
  %298 = load <2 x double>, ptr %297, align 8, !tbaa !10
  %299 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %186, <2 x double> %298, <2 x double> %296)
  %300 = getelementptr i8, ptr %215, i64 9512
  %301 = load <2 x double>, ptr %300, align 8, !tbaa !10
  %302 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %188, <2 x double> %301, <2 x double> %299)
  %303 = getelementptr i8, ptr %215, i64 9840
  %304 = load <2 x double>, ptr %303, align 8, !tbaa !10
  %305 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %190, <2 x double> %304, <2 x double> %302)
  %306 = getelementptr i8, ptr %215, i64 10168
  %307 = load <2 x double>, ptr %306, align 8, !tbaa !10
  %308 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %192, <2 x double> %307, <2 x double> %305)
  %309 = getelementptr i8, ptr %215, i64 10496
  %310 = load <2 x double>, ptr %309, align 8, !tbaa !10
  %311 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %194, <2 x double> %310, <2 x double> %308)
  %312 = getelementptr i8, ptr %215, i64 10824
  %313 = load <2 x double>, ptr %312, align 8, !tbaa !10
  %314 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %196, <2 x double> %313, <2 x double> %311)
  %315 = getelementptr i8, ptr %215, i64 11152
  %316 = load <2 x double>, ptr %315, align 8, !tbaa !10
  %317 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %198, <2 x double> %316, <2 x double> %314)
  %318 = getelementptr i8, ptr %215, i64 11480
  %319 = load <2 x double>, ptr %318, align 8, !tbaa !10
  %320 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %200, <2 x double> %319, <2 x double> %317)
  %321 = getelementptr i8, ptr %215, i64 11808
  %322 = load <2 x double>, ptr %321, align 8, !tbaa !10
  %323 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %202, <2 x double> %322, <2 x double> %320)
  %324 = getelementptr i8, ptr %215, i64 12136
  %325 = load <2 x double>, ptr %324, align 8, !tbaa !10
  %326 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %204, <2 x double> %325, <2 x double> %323)
  %327 = getelementptr i8, ptr %215, i64 12464
  %328 = load <2 x double>, ptr %327, align 8, !tbaa !10
  %329 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %206, <2 x double> %328, <2 x double> %326)
  %330 = getelementptr i8, ptr %215, i64 12792
  %331 = load <2 x double>, ptr %330, align 8, !tbaa !10
  %332 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %208, <2 x double> %331, <2 x double> %329)
  %333 = getelementptr i8, ptr %215, i64 13120
  %334 = load <2 x double>, ptr %333, align 8, !tbaa !10
  %335 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %210, <2 x double> %334, <2 x double> %332)
  store <2 x double> %335, ptr %214, align 8, !tbaa !10
  %336 = add nuw i64 %212, 2
  %337 = icmp eq i64 %336, 40
  br i1 %337, label %338, label %211, !llvm.loop !15

338:                                              ; preds = %211
  %339 = add nuw nsw i64 %48, 1
  %340 = icmp eq i64 %339, 41
  br i1 %340, label %341, label %47, !llvm.loop !18

341:                                              ; preds = %338
  %342 = add nsw i32 %0, 1
  %343 = sext i32 %342 to i64
  %344 = getelementptr inbounds [41 x double], ptr @rmr, i64 %343
  %345 = getelementptr inbounds double, ptr %344, i64 %343
  %346 = load double, ptr %345, align 8, !tbaa !10
  %347 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %346)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #6

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  tail call void @Mm(i32 noundef 0)
  tail call void @Mm(i32 noundef 1)
  tail call void @Mm(i32 noundef 2)
  tail call void @Mm(i32 noundef 3)
  tail call void @Mm(i32 noundef 4)
  tail call void @Mm(i32 noundef 5)
  tail call void @Mm(i32 noundef 6)
  tail call void @Mm(i32 noundef 7)
  tail call void @Mm(i32 noundef 8)
  tail call void @Mm(i32 noundef 9)
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #7

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
!11 = !{!"double", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = distinct !{!15, !13, !16, !17}
!16 = !{!"llvm.loop.isvectorized", i32 1}
!17 = !{!"llvm.loop.unroll.runtime.disable"}
!18 = distinct !{!18, !13}
