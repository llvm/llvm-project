; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/IntMM.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/IntMM.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.element = type { i32, i32 }
%struct.complex = type { float, float }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@ima = dso_local global [41 x [41 x i32]] zeroinitializer, align 4
@imb = dso_local global [41 x [41 x i32]] zeroinitializer, align 4
@imr = dso_local global [41 x [41 x i32]] zeroinitializer, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
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
@rma = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmb = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmr = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
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
define dso_local void @Initmatrix(ptr noundef writeonly captures(none) %0) local_unnamed_addr #2 {
  %2 = load i64, ptr @seed, align 8, !tbaa !6
  %3 = freeze i64 %2
  br label %4

4:                                                ; preds = %1, %21
  %5 = phi i64 [ 1, %1 ], [ %22, %21 ]
  %6 = phi i64 [ %3, %1 ], [ %13, %21 ]
  %7 = getelementptr inbounds nuw [41 x i32], ptr %0, i64 %5
  br label %8

8:                                                ; preds = %4, %8
  %9 = phi i64 [ 1, %4 ], [ %19, %8 ]
  %10 = phi i64 [ %6, %4 ], [ %13, %8 ]
  %11 = mul i64 %10, 1309
  %12 = add i64 %11, 13849
  %13 = and i64 %12, 65535
  %14 = trunc i64 %12 to i16
  %15 = urem i16 %14, 120
  %16 = zext nneg i16 %15 to i32
  %17 = add nsw i32 %16, -60
  %18 = getelementptr inbounds nuw i32, ptr %7, i64 %9
  store i32 %17, ptr %18, align 4, !tbaa !10
  %19 = add nuw nsw i64 %9, 1
  %20 = icmp eq i64 %19, 41
  br i1 %20, label %21, label %8, !llvm.loop !12

21:                                               ; preds = %8
  %22 = add nuw nsw i64 %5, 1
  %23 = icmp eq i64 %22, 41
  br i1 %23, label %24, label %4, !llvm.loop !14

24:                                               ; preds = %21
  store i64 %13, ptr @seed, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @Innerproduct(ptr noundef writeonly captures(none) initializes((0, 4)) %0, ptr noundef readonly captures(none) %1, ptr noundef readonly captures(none) %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #3 {
  store i32 0, ptr %0, align 4, !tbaa !10
  %6 = sext i32 %3 to i64
  %7 = getelementptr inbounds [41 x i32], ptr %1, i64 %6
  %8 = sext i32 %4 to i64
  %9 = getelementptr i32, ptr %2, i64 %8
  %10 = getelementptr inbounds nuw i8, ptr %7, i64 4
  %11 = load i32, ptr %10, align 4, !tbaa !10
  %12 = getelementptr i8, ptr %9, i64 164
  %13 = load i32, ptr %12, align 4, !tbaa !10
  %14 = mul nsw i32 %13, %11
  store i32 %14, ptr %0, align 4, !tbaa !10
  %15 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %16 = load i32, ptr %15, align 4, !tbaa !10
  %17 = getelementptr i8, ptr %9, i64 328
  %18 = load i32, ptr %17, align 4, !tbaa !10
  %19 = mul nsw i32 %18, %16
  %20 = add nsw i32 %19, %14
  store i32 %20, ptr %0, align 4, !tbaa !10
  %21 = getelementptr inbounds nuw i8, ptr %7, i64 12
  %22 = load i32, ptr %21, align 4, !tbaa !10
  %23 = getelementptr i8, ptr %9, i64 492
  %24 = load i32, ptr %23, align 4, !tbaa !10
  %25 = mul nsw i32 %24, %22
  %26 = add nsw i32 %25, %20
  store i32 %26, ptr %0, align 4, !tbaa !10
  %27 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %28 = load i32, ptr %27, align 4, !tbaa !10
  %29 = getelementptr i8, ptr %9, i64 656
  %30 = load i32, ptr %29, align 4, !tbaa !10
  %31 = mul nsw i32 %30, %28
  %32 = add nsw i32 %31, %26
  store i32 %32, ptr %0, align 4, !tbaa !10
  %33 = getelementptr inbounds nuw i8, ptr %7, i64 20
  %34 = load i32, ptr %33, align 4, !tbaa !10
  %35 = getelementptr i8, ptr %9, i64 820
  %36 = load i32, ptr %35, align 4, !tbaa !10
  %37 = mul nsw i32 %36, %34
  %38 = add nsw i32 %37, %32
  store i32 %38, ptr %0, align 4, !tbaa !10
  %39 = getelementptr inbounds nuw i8, ptr %7, i64 24
  %40 = load i32, ptr %39, align 4, !tbaa !10
  %41 = getelementptr i8, ptr %9, i64 984
  %42 = load i32, ptr %41, align 4, !tbaa !10
  %43 = mul nsw i32 %42, %40
  %44 = add nsw i32 %43, %38
  store i32 %44, ptr %0, align 4, !tbaa !10
  %45 = getelementptr inbounds nuw i8, ptr %7, i64 28
  %46 = load i32, ptr %45, align 4, !tbaa !10
  %47 = getelementptr i8, ptr %9, i64 1148
  %48 = load i32, ptr %47, align 4, !tbaa !10
  %49 = mul nsw i32 %48, %46
  %50 = add nsw i32 %49, %44
  store i32 %50, ptr %0, align 4, !tbaa !10
  %51 = getelementptr inbounds nuw i8, ptr %7, i64 32
  %52 = load i32, ptr %51, align 4, !tbaa !10
  %53 = getelementptr i8, ptr %9, i64 1312
  %54 = load i32, ptr %53, align 4, !tbaa !10
  %55 = mul nsw i32 %54, %52
  %56 = add nsw i32 %55, %50
  store i32 %56, ptr %0, align 4, !tbaa !10
  %57 = getelementptr inbounds nuw i8, ptr %7, i64 36
  %58 = load i32, ptr %57, align 4, !tbaa !10
  %59 = getelementptr i8, ptr %9, i64 1476
  %60 = load i32, ptr %59, align 4, !tbaa !10
  %61 = mul nsw i32 %60, %58
  %62 = add nsw i32 %61, %56
  store i32 %62, ptr %0, align 4, !tbaa !10
  %63 = getelementptr inbounds nuw i8, ptr %7, i64 40
  %64 = load i32, ptr %63, align 4, !tbaa !10
  %65 = getelementptr i8, ptr %9, i64 1640
  %66 = load i32, ptr %65, align 4, !tbaa !10
  %67 = mul nsw i32 %66, %64
  %68 = add nsw i32 %67, %62
  store i32 %68, ptr %0, align 4, !tbaa !10
  %69 = getelementptr inbounds nuw i8, ptr %7, i64 44
  %70 = load i32, ptr %69, align 4, !tbaa !10
  %71 = getelementptr i8, ptr %9, i64 1804
  %72 = load i32, ptr %71, align 4, !tbaa !10
  %73 = mul nsw i32 %72, %70
  %74 = add nsw i32 %73, %68
  store i32 %74, ptr %0, align 4, !tbaa !10
  %75 = getelementptr inbounds nuw i8, ptr %7, i64 48
  %76 = load i32, ptr %75, align 4, !tbaa !10
  %77 = getelementptr i8, ptr %9, i64 1968
  %78 = load i32, ptr %77, align 4, !tbaa !10
  %79 = mul nsw i32 %78, %76
  %80 = add nsw i32 %79, %74
  store i32 %80, ptr %0, align 4, !tbaa !10
  %81 = getelementptr inbounds nuw i8, ptr %7, i64 52
  %82 = load i32, ptr %81, align 4, !tbaa !10
  %83 = getelementptr i8, ptr %9, i64 2132
  %84 = load i32, ptr %83, align 4, !tbaa !10
  %85 = mul nsw i32 %84, %82
  %86 = add nsw i32 %85, %80
  store i32 %86, ptr %0, align 4, !tbaa !10
  %87 = getelementptr inbounds nuw i8, ptr %7, i64 56
  %88 = load i32, ptr %87, align 4, !tbaa !10
  %89 = getelementptr i8, ptr %9, i64 2296
  %90 = load i32, ptr %89, align 4, !tbaa !10
  %91 = mul nsw i32 %90, %88
  %92 = add nsw i32 %91, %86
  store i32 %92, ptr %0, align 4, !tbaa !10
  %93 = getelementptr inbounds nuw i8, ptr %7, i64 60
  %94 = load i32, ptr %93, align 4, !tbaa !10
  %95 = getelementptr i8, ptr %9, i64 2460
  %96 = load i32, ptr %95, align 4, !tbaa !10
  %97 = mul nsw i32 %96, %94
  %98 = add nsw i32 %97, %92
  store i32 %98, ptr %0, align 4, !tbaa !10
  %99 = getelementptr inbounds nuw i8, ptr %7, i64 64
  %100 = load i32, ptr %99, align 4, !tbaa !10
  %101 = getelementptr i8, ptr %9, i64 2624
  %102 = load i32, ptr %101, align 4, !tbaa !10
  %103 = mul nsw i32 %102, %100
  %104 = add nsw i32 %103, %98
  store i32 %104, ptr %0, align 4, !tbaa !10
  %105 = getelementptr inbounds nuw i8, ptr %7, i64 68
  %106 = load i32, ptr %105, align 4, !tbaa !10
  %107 = getelementptr i8, ptr %9, i64 2788
  %108 = load i32, ptr %107, align 4, !tbaa !10
  %109 = mul nsw i32 %108, %106
  %110 = add nsw i32 %109, %104
  store i32 %110, ptr %0, align 4, !tbaa !10
  %111 = getelementptr inbounds nuw i8, ptr %7, i64 72
  %112 = load i32, ptr %111, align 4, !tbaa !10
  %113 = getelementptr i8, ptr %9, i64 2952
  %114 = load i32, ptr %113, align 4, !tbaa !10
  %115 = mul nsw i32 %114, %112
  %116 = add nsw i32 %115, %110
  store i32 %116, ptr %0, align 4, !tbaa !10
  %117 = getelementptr inbounds nuw i8, ptr %7, i64 76
  %118 = load i32, ptr %117, align 4, !tbaa !10
  %119 = getelementptr i8, ptr %9, i64 3116
  %120 = load i32, ptr %119, align 4, !tbaa !10
  %121 = mul nsw i32 %120, %118
  %122 = add nsw i32 %121, %116
  store i32 %122, ptr %0, align 4, !tbaa !10
  %123 = getelementptr inbounds nuw i8, ptr %7, i64 80
  %124 = load i32, ptr %123, align 4, !tbaa !10
  %125 = getelementptr i8, ptr %9, i64 3280
  %126 = load i32, ptr %125, align 4, !tbaa !10
  %127 = mul nsw i32 %126, %124
  %128 = add nsw i32 %127, %122
  store i32 %128, ptr %0, align 4, !tbaa !10
  %129 = getelementptr inbounds nuw i8, ptr %7, i64 84
  %130 = load i32, ptr %129, align 4, !tbaa !10
  %131 = getelementptr i8, ptr %9, i64 3444
  %132 = load i32, ptr %131, align 4, !tbaa !10
  %133 = mul nsw i32 %132, %130
  %134 = add nsw i32 %133, %128
  store i32 %134, ptr %0, align 4, !tbaa !10
  %135 = getelementptr inbounds nuw i8, ptr %7, i64 88
  %136 = load i32, ptr %135, align 4, !tbaa !10
  %137 = getelementptr i8, ptr %9, i64 3608
  %138 = load i32, ptr %137, align 4, !tbaa !10
  %139 = mul nsw i32 %138, %136
  %140 = add nsw i32 %139, %134
  store i32 %140, ptr %0, align 4, !tbaa !10
  %141 = getelementptr inbounds nuw i8, ptr %7, i64 92
  %142 = load i32, ptr %141, align 4, !tbaa !10
  %143 = getelementptr i8, ptr %9, i64 3772
  %144 = load i32, ptr %143, align 4, !tbaa !10
  %145 = mul nsw i32 %144, %142
  %146 = add nsw i32 %145, %140
  store i32 %146, ptr %0, align 4, !tbaa !10
  %147 = getelementptr inbounds nuw i8, ptr %7, i64 96
  %148 = load i32, ptr %147, align 4, !tbaa !10
  %149 = getelementptr i8, ptr %9, i64 3936
  %150 = load i32, ptr %149, align 4, !tbaa !10
  %151 = mul nsw i32 %150, %148
  %152 = add nsw i32 %151, %146
  store i32 %152, ptr %0, align 4, !tbaa !10
  %153 = getelementptr inbounds nuw i8, ptr %7, i64 100
  %154 = load i32, ptr %153, align 4, !tbaa !10
  %155 = getelementptr i8, ptr %9, i64 4100
  %156 = load i32, ptr %155, align 4, !tbaa !10
  %157 = mul nsw i32 %156, %154
  %158 = add nsw i32 %157, %152
  store i32 %158, ptr %0, align 4, !tbaa !10
  %159 = getelementptr inbounds nuw i8, ptr %7, i64 104
  %160 = load i32, ptr %159, align 4, !tbaa !10
  %161 = getelementptr i8, ptr %9, i64 4264
  %162 = load i32, ptr %161, align 4, !tbaa !10
  %163 = mul nsw i32 %162, %160
  %164 = add nsw i32 %163, %158
  store i32 %164, ptr %0, align 4, !tbaa !10
  %165 = getelementptr inbounds nuw i8, ptr %7, i64 108
  %166 = load i32, ptr %165, align 4, !tbaa !10
  %167 = getelementptr i8, ptr %9, i64 4428
  %168 = load i32, ptr %167, align 4, !tbaa !10
  %169 = mul nsw i32 %168, %166
  %170 = add nsw i32 %169, %164
  store i32 %170, ptr %0, align 4, !tbaa !10
  %171 = getelementptr inbounds nuw i8, ptr %7, i64 112
  %172 = load i32, ptr %171, align 4, !tbaa !10
  %173 = getelementptr i8, ptr %9, i64 4592
  %174 = load i32, ptr %173, align 4, !tbaa !10
  %175 = mul nsw i32 %174, %172
  %176 = add nsw i32 %175, %170
  store i32 %176, ptr %0, align 4, !tbaa !10
  %177 = getelementptr inbounds nuw i8, ptr %7, i64 116
  %178 = load i32, ptr %177, align 4, !tbaa !10
  %179 = getelementptr i8, ptr %9, i64 4756
  %180 = load i32, ptr %179, align 4, !tbaa !10
  %181 = mul nsw i32 %180, %178
  %182 = add nsw i32 %181, %176
  store i32 %182, ptr %0, align 4, !tbaa !10
  %183 = getelementptr inbounds nuw i8, ptr %7, i64 120
  %184 = load i32, ptr %183, align 4, !tbaa !10
  %185 = getelementptr i8, ptr %9, i64 4920
  %186 = load i32, ptr %185, align 4, !tbaa !10
  %187 = mul nsw i32 %186, %184
  %188 = add nsw i32 %187, %182
  store i32 %188, ptr %0, align 4, !tbaa !10
  %189 = getelementptr inbounds nuw i8, ptr %7, i64 124
  %190 = load i32, ptr %189, align 4, !tbaa !10
  %191 = getelementptr i8, ptr %9, i64 5084
  %192 = load i32, ptr %191, align 4, !tbaa !10
  %193 = mul nsw i32 %192, %190
  %194 = add nsw i32 %193, %188
  store i32 %194, ptr %0, align 4, !tbaa !10
  %195 = getelementptr inbounds nuw i8, ptr %7, i64 128
  %196 = load i32, ptr %195, align 4, !tbaa !10
  %197 = getelementptr i8, ptr %9, i64 5248
  %198 = load i32, ptr %197, align 4, !tbaa !10
  %199 = mul nsw i32 %198, %196
  %200 = add nsw i32 %199, %194
  store i32 %200, ptr %0, align 4, !tbaa !10
  %201 = getelementptr inbounds nuw i8, ptr %7, i64 132
  %202 = load i32, ptr %201, align 4, !tbaa !10
  %203 = getelementptr i8, ptr %9, i64 5412
  %204 = load i32, ptr %203, align 4, !tbaa !10
  %205 = mul nsw i32 %204, %202
  %206 = add nsw i32 %205, %200
  store i32 %206, ptr %0, align 4, !tbaa !10
  %207 = getelementptr inbounds nuw i8, ptr %7, i64 136
  %208 = load i32, ptr %207, align 4, !tbaa !10
  %209 = getelementptr i8, ptr %9, i64 5576
  %210 = load i32, ptr %209, align 4, !tbaa !10
  %211 = mul nsw i32 %210, %208
  %212 = add nsw i32 %211, %206
  store i32 %212, ptr %0, align 4, !tbaa !10
  %213 = getelementptr inbounds nuw i8, ptr %7, i64 140
  %214 = load i32, ptr %213, align 4, !tbaa !10
  %215 = getelementptr i8, ptr %9, i64 5740
  %216 = load i32, ptr %215, align 4, !tbaa !10
  %217 = mul nsw i32 %216, %214
  %218 = add nsw i32 %217, %212
  store i32 %218, ptr %0, align 4, !tbaa !10
  %219 = getelementptr inbounds nuw i8, ptr %7, i64 144
  %220 = load i32, ptr %219, align 4, !tbaa !10
  %221 = getelementptr i8, ptr %9, i64 5904
  %222 = load i32, ptr %221, align 4, !tbaa !10
  %223 = mul nsw i32 %222, %220
  %224 = add nsw i32 %223, %218
  store i32 %224, ptr %0, align 4, !tbaa !10
  %225 = getelementptr inbounds nuw i8, ptr %7, i64 148
  %226 = load i32, ptr %225, align 4, !tbaa !10
  %227 = getelementptr i8, ptr %9, i64 6068
  %228 = load i32, ptr %227, align 4, !tbaa !10
  %229 = mul nsw i32 %228, %226
  %230 = add nsw i32 %229, %224
  store i32 %230, ptr %0, align 4, !tbaa !10
  %231 = getelementptr inbounds nuw i8, ptr %7, i64 152
  %232 = load i32, ptr %231, align 4, !tbaa !10
  %233 = getelementptr i8, ptr %9, i64 6232
  %234 = load i32, ptr %233, align 4, !tbaa !10
  %235 = mul nsw i32 %234, %232
  %236 = add nsw i32 %235, %230
  store i32 %236, ptr %0, align 4, !tbaa !10
  %237 = getelementptr inbounds nuw i8, ptr %7, i64 156
  %238 = load i32, ptr %237, align 4, !tbaa !10
  %239 = getelementptr i8, ptr %9, i64 6396
  %240 = load i32, ptr %239, align 4, !tbaa !10
  %241 = mul nsw i32 %240, %238
  %242 = add nsw i32 %241, %236
  store i32 %242, ptr %0, align 4, !tbaa !10
  %243 = getelementptr inbounds nuw i8, ptr %7, i64 160
  %244 = load i32, ptr %243, align 4, !tbaa !10
  %245 = getelementptr i8, ptr %9, i64 6560
  %246 = load i32, ptr %245, align 4, !tbaa !10
  %247 = mul nsw i32 %246, %244
  %248 = add nsw i32 %247, %242
  store i32 %248, ptr %0, align 4, !tbaa !10
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Intmm(i32 noundef %0) local_unnamed_addr #4 {
  br label %2

2:                                                ; preds = %19, %1
  %3 = phi i64 [ 1, %1 ], [ %20, %19 ]
  %4 = phi i64 [ 74755, %1 ], [ %11, %19 ]
  %5 = getelementptr inbounds nuw [41 x i32], ptr @ima, i64 %3
  br label %6

6:                                                ; preds = %6, %2
  %7 = phi i64 [ 1, %2 ], [ %17, %6 ]
  %8 = phi i64 [ %4, %2 ], [ %11, %6 ]
  %9 = mul nuw nsw i64 %8, 1309
  %10 = add nuw nsw i64 %9, 13849
  %11 = and i64 %10, 65535
  %12 = trunc i64 %10 to i16
  %13 = urem i16 %12, 120
  %14 = zext nneg i16 %13 to i32
  %15 = add nsw i32 %14, -60
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %7
  store i32 %15, ptr %16, align 4, !tbaa !10
  %17 = add nuw nsw i64 %7, 1
  %18 = icmp eq i64 %17, 41
  br i1 %18, label %19, label %6, !llvm.loop !12

19:                                               ; preds = %6
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp eq i64 %20, 41
  br i1 %21, label %22, label %2, !llvm.loop !14

22:                                               ; preds = %19, %39
  %23 = phi i64 [ %40, %39 ], [ 1, %19 ]
  %24 = phi i64 [ %31, %39 ], [ %11, %19 ]
  %25 = getelementptr inbounds nuw [41 x i32], ptr @imb, i64 %23
  br label %26

26:                                               ; preds = %26, %22
  %27 = phi i64 [ 1, %22 ], [ %37, %26 ]
  %28 = phi i64 [ %24, %22 ], [ %31, %26 ]
  %29 = mul nuw nsw i64 %28, 1309
  %30 = add nuw nsw i64 %29, 13849
  %31 = and i64 %30, 65535
  %32 = trunc i64 %30 to i16
  %33 = urem i16 %32, 120
  %34 = zext nneg i16 %33 to i32
  %35 = add nsw i32 %34, -60
  %36 = getelementptr inbounds nuw i32, ptr %25, i64 %27
  store i32 %35, ptr %36, align 4, !tbaa !10
  %37 = add nuw nsw i64 %27, 1
  %38 = icmp eq i64 %37, 41
  br i1 %38, label %39, label %26, !llvm.loop !12

39:                                               ; preds = %26
  %40 = add nuw nsw i64 %23, 1
  %41 = icmp eq i64 %40, 41
  br i1 %41, label %42, label %22, !llvm.loop !14

42:                                               ; preds = %39
  store i64 %31, ptr @seed, align 8, !tbaa !6
  br label %43

43:                                               ; preds = %42, %43
  %44 = phi i64 [ 1, %42 ], [ %87, %43 ]
  %45 = getelementptr inbounds nuw [41 x i32], ptr @imr, i64 %44
  %46 = getelementptr inbounds nuw i8, ptr %45, i64 4
  %47 = trunc nuw nsw i64 %44 to i32
  tail call void @Innerproduct(ptr noundef nonnull %46, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 1)
  %48 = getelementptr inbounds nuw i8, ptr %45, i64 8
  tail call void @Innerproduct(ptr noundef nonnull %48, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 2)
  %49 = getelementptr inbounds nuw i8, ptr %45, i64 12
  tail call void @Innerproduct(ptr noundef nonnull %49, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 3)
  %50 = getelementptr inbounds nuw i8, ptr %45, i64 16
  tail call void @Innerproduct(ptr noundef nonnull %50, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 4)
  %51 = getelementptr inbounds nuw i8, ptr %45, i64 20
  tail call void @Innerproduct(ptr noundef nonnull %51, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 5)
  %52 = getelementptr inbounds nuw i8, ptr %45, i64 24
  tail call void @Innerproduct(ptr noundef nonnull %52, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 6)
  %53 = getelementptr inbounds nuw i8, ptr %45, i64 28
  tail call void @Innerproduct(ptr noundef nonnull %53, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 7)
  %54 = getelementptr inbounds nuw i8, ptr %45, i64 32
  tail call void @Innerproduct(ptr noundef nonnull %54, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 8)
  %55 = getelementptr inbounds nuw i8, ptr %45, i64 36
  tail call void @Innerproduct(ptr noundef nonnull %55, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 9)
  %56 = getelementptr inbounds nuw i8, ptr %45, i64 40
  tail call void @Innerproduct(ptr noundef nonnull %56, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 10)
  %57 = getelementptr inbounds nuw i8, ptr %45, i64 44
  tail call void @Innerproduct(ptr noundef nonnull %57, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 11)
  %58 = getelementptr inbounds nuw i8, ptr %45, i64 48
  tail call void @Innerproduct(ptr noundef nonnull %58, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 12)
  %59 = getelementptr inbounds nuw i8, ptr %45, i64 52
  tail call void @Innerproduct(ptr noundef nonnull %59, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 13)
  %60 = getelementptr inbounds nuw i8, ptr %45, i64 56
  tail call void @Innerproduct(ptr noundef nonnull %60, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 14)
  %61 = getelementptr inbounds nuw i8, ptr %45, i64 60
  tail call void @Innerproduct(ptr noundef nonnull %61, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 15)
  %62 = getelementptr inbounds nuw i8, ptr %45, i64 64
  tail call void @Innerproduct(ptr noundef nonnull %62, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 16)
  %63 = getelementptr inbounds nuw i8, ptr %45, i64 68
  tail call void @Innerproduct(ptr noundef nonnull %63, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 17)
  %64 = getelementptr inbounds nuw i8, ptr %45, i64 72
  tail call void @Innerproduct(ptr noundef nonnull %64, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 18)
  %65 = getelementptr inbounds nuw i8, ptr %45, i64 76
  tail call void @Innerproduct(ptr noundef nonnull %65, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 19)
  %66 = getelementptr inbounds nuw i8, ptr %45, i64 80
  tail call void @Innerproduct(ptr noundef nonnull %66, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 20)
  %67 = getelementptr inbounds nuw i8, ptr %45, i64 84
  tail call void @Innerproduct(ptr noundef nonnull %67, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 21)
  %68 = getelementptr inbounds nuw i8, ptr %45, i64 88
  tail call void @Innerproduct(ptr noundef nonnull %68, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 22)
  %69 = getelementptr inbounds nuw i8, ptr %45, i64 92
  tail call void @Innerproduct(ptr noundef nonnull %69, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 23)
  %70 = getelementptr inbounds nuw i8, ptr %45, i64 96
  tail call void @Innerproduct(ptr noundef nonnull %70, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 24)
  %71 = getelementptr inbounds nuw i8, ptr %45, i64 100
  tail call void @Innerproduct(ptr noundef nonnull %71, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 25)
  %72 = getelementptr inbounds nuw i8, ptr %45, i64 104
  tail call void @Innerproduct(ptr noundef nonnull %72, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 26)
  %73 = getelementptr inbounds nuw i8, ptr %45, i64 108
  tail call void @Innerproduct(ptr noundef nonnull %73, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 27)
  %74 = getelementptr inbounds nuw i8, ptr %45, i64 112
  tail call void @Innerproduct(ptr noundef nonnull %74, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 28)
  %75 = getelementptr inbounds nuw i8, ptr %45, i64 116
  tail call void @Innerproduct(ptr noundef nonnull %75, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 29)
  %76 = getelementptr inbounds nuw i8, ptr %45, i64 120
  tail call void @Innerproduct(ptr noundef nonnull %76, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 30)
  %77 = getelementptr inbounds nuw i8, ptr %45, i64 124
  tail call void @Innerproduct(ptr noundef nonnull %77, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 31)
  %78 = getelementptr inbounds nuw i8, ptr %45, i64 128
  tail call void @Innerproduct(ptr noundef nonnull %78, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 32)
  %79 = getelementptr inbounds nuw i8, ptr %45, i64 132
  tail call void @Innerproduct(ptr noundef nonnull %79, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 33)
  %80 = getelementptr inbounds nuw i8, ptr %45, i64 136
  tail call void @Innerproduct(ptr noundef nonnull %80, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 34)
  %81 = getelementptr inbounds nuw i8, ptr %45, i64 140
  tail call void @Innerproduct(ptr noundef nonnull %81, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 35)
  %82 = getelementptr inbounds nuw i8, ptr %45, i64 144
  tail call void @Innerproduct(ptr noundef nonnull %82, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 36)
  %83 = getelementptr inbounds nuw i8, ptr %45, i64 148
  tail call void @Innerproduct(ptr noundef nonnull %83, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 37)
  %84 = getelementptr inbounds nuw i8, ptr %45, i64 152
  tail call void @Innerproduct(ptr noundef nonnull %84, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 38)
  %85 = getelementptr inbounds nuw i8, ptr %45, i64 156
  tail call void @Innerproduct(ptr noundef nonnull %85, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 39)
  %86 = getelementptr inbounds nuw i8, ptr %45, i64 160
  tail call void @Innerproduct(ptr noundef nonnull %86, ptr noundef nonnull @ima, ptr noundef nonnull @imb, i32 noundef %47, i32 noundef 40)
  %87 = add nuw nsw i64 %44, 1
  %88 = icmp eq i64 %87, 41
  br i1 %88, label %89, label %43, !llvm.loop !15

89:                                               ; preds = %43
  %90 = add nsw i32 %0, 1
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds [41 x i32], ptr @imr, i64 %91
  %93 = getelementptr inbounds i32, ptr %92, i64 %91
  %94 = load i32, ptr %93, align 4, !tbaa !10
  %95 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %94)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #5

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  tail call void @Intmm(i32 noundef 0)
  tail call void @Intmm(i32 noundef 1)
  tail call void @Intmm(i32 noundef 2)
  tail call void @Intmm(i32 noundef 3)
  tail call void @Intmm(i32 noundef 4)
  tail call void @Intmm(i32 noundef 5)
  tail call void @Intmm(i32 noundef 6)
  tail call void @Intmm(i32 noundef 7)
  tail call void @Intmm(i32 noundef 8)
  tail call void @Intmm(i32 noundef 9)
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree norecurse nosync nounwind memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = distinct !{!15, !13}
