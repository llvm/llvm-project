// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @intrinsics
llvm.func @intrinsics(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: !llvm.ptr<i8>) {
  %c3 = llvm.mlir.constant(3 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call float @llvm.fmuladd.f32
  "llvm.intr.fmuladd"(%arg0, %arg1, %arg0) : (f32, f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.fmuladd.v8f32
  "llvm.intr.fmuladd"(%arg2, %arg2, %arg2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK: call float @llvm.fma.f32
  "llvm.intr.fma"(%arg0, %arg1, %arg0) : (f32, f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.fma.v8f32
  "llvm.intr.fma"(%arg2, %arg2, %arg2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK: call void @llvm.prefetch.p0i8(i8* %3, i32 0, i32 3, i32 1)
  "llvm.intr.prefetch"(%arg3, %c0, %c3, %c1) : (!llvm.ptr<i8>, i32, i32, i32) -> ()
  llvm.return
}

// CHECK-LABEL: @exp_test
llvm.func @exp_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.exp.f32
  "llvm.intr.exp"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.exp.v8f32
  "llvm.intr.exp"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @exp2_test
llvm.func @exp2_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.exp2.f32
  "llvm.intr.exp2"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.exp2.v8f32
  "llvm.intr.exp2"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @log_test
llvm.func @log_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.log.f32
  "llvm.intr.log"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.log.v8f32
  "llvm.intr.log"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @log10_test
llvm.func @log10_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.log10.f32
  "llvm.intr.log10"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.log10.v8f32
  "llvm.intr.log10"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @log2_test
llvm.func @log2_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.log2.f32
  "llvm.intr.log2"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.log2.v8f32
  "llvm.intr.log2"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @fabs_test
llvm.func @fabs_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.fabs.f32
  "llvm.intr.fabs"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.fabs.v8f32
  "llvm.intr.fabs"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @sqrt_test
llvm.func @sqrt_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.sqrt.f32
  "llvm.intr.sqrt"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.sqrt.v8f32
  "llvm.intr.sqrt"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @ceil_test
llvm.func @ceil_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.ceil.f32
  "llvm.intr.ceil"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.ceil.v8f32
  "llvm.intr.ceil"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @floor_test
llvm.func @floor_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.floor.f32
  "llvm.intr.floor"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.floor.v8f32
  "llvm.intr.floor"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @cos_test
llvm.func @cos_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.cos.f32
  "llvm.intr.cos"(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.cos.v8f32
  "llvm.intr.cos"(%arg1) : (vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @copysign_test
llvm.func @copysign_test(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: vector<8xf32>) {
  // CHECK: call float @llvm.copysign.f32
  "llvm.intr.copysign"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.copysign.v8f32
  "llvm.intr.copysign"(%arg2, %arg3) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @pow_test
llvm.func @pow_test(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: vector<8xf32>) {
  // CHECK: call float @llvm.pow.f32
  "llvm.intr.pow"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.pow.v8f32
  "llvm.intr.pow"(%arg2, %arg3) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @bitreverse_test
llvm.func @bitreverse_test(%arg0: i32, %arg1: vector<8xi32>) {
  // CHECK: call i32 @llvm.bitreverse.i32
  "llvm.intr.bitreverse"(%arg0) : (i32) -> i32
  // CHECK: call <8 x i32> @llvm.bitreverse.v8i32
  "llvm.intr.bitreverse"(%arg1) : (vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @ctpop_test
llvm.func @ctpop_test(%arg0: i32, %arg1: vector<8xi32>) {
  // CHECK: call i32 @llvm.ctpop.i32
  "llvm.intr.ctpop"(%arg0) : (i32) -> i32
  // CHECK: call <8 x i32> @llvm.ctpop.v8i32
  "llvm.intr.ctpop"(%arg1) : (vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @maximum_test
llvm.func @maximum_test(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: vector<8xf32>) {
  // CHECK: call float @llvm.maximum.f32
  "llvm.intr.maximum"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.maximum.v8f32
  "llvm.intr.maximum"(%arg2, %arg3) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @minimum_test
llvm.func @minimum_test(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: vector<8xf32>) {
  // CHECK: call float @llvm.minimum.f32
  "llvm.intr.minimum"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.minimum.v8f32
  "llvm.intr.minimum"(%arg2, %arg3) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @maxnum_test
llvm.func @maxnum_test(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: vector<8xf32>) {
  // CHECK: call float @llvm.maxnum.f32
  "llvm.intr.maxnum"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.maxnum.v8f32
  "llvm.intr.maxnum"(%arg2, %arg3) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @minnum_test
llvm.func @minnum_test(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: vector<8xf32>) {
  // CHECK: call float @llvm.minnum.f32
  "llvm.intr.minnum"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.minnum.v8f32
  "llvm.intr.minnum"(%arg2, %arg3) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// CHECK-LABEL: @smax_test
llvm.func @smax_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.smax.i32
  "llvm.intr.smax"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.smax.v8i32
  "llvm.intr.smax"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @smin_test
llvm.func @smin_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.smin.i32
  "llvm.intr.smin"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.smin.v8i32
  "llvm.intr.smin"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @vector_reductions
llvm.func @vector_reductions(%arg0: f32, %arg1: vector<8xf32>, %arg2: vector<8xi32>) {
  // CHECK: call i32 @llvm.vector.reduce.add.v8i32
  "llvm.intr.vector.reduce.add"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call i32 @llvm.vector.reduce.and.v8i32
  "llvm.intr.vector.reduce.and"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call float @llvm.vector.reduce.fmax.v8f32
  "llvm.intr.vector.reduce.fmax"(%arg1) : (vector<8xf32>) -> f32
  // CHECK: call float @llvm.vector.reduce.fmin.v8f32
  "llvm.intr.vector.reduce.fmin"(%arg1) : (vector<8xf32>) -> f32
  // CHECK: call i32 @llvm.vector.reduce.mul.v8i32
  "llvm.intr.vector.reduce.mul"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call i32 @llvm.vector.reduce.or.v8i32
  "llvm.intr.vector.reduce.or"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call i32 @llvm.vector.reduce.smax.v8i32
  "llvm.intr.vector.reduce.smax"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call i32 @llvm.vector.reduce.smin.v8i32
  "llvm.intr.vector.reduce.smin"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call i32 @llvm.vector.reduce.umax.v8i32
  "llvm.intr.vector.reduce.umax"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call i32 @llvm.vector.reduce.umin.v8i32
  "llvm.intr.vector.reduce.umin"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call float @llvm.vector.reduce.fadd.v8f32
  "llvm.intr.vector.reduce.fadd"(%arg0, %arg1) : (f32, vector<8xf32>) -> f32
  // CHECK: call float @llvm.vector.reduce.fmul.v8f32
  "llvm.intr.vector.reduce.fmul"(%arg0, %arg1) : (f32, vector<8xf32>) -> f32
  // CHECK: call reassoc float @llvm.vector.reduce.fadd.v8f32
  "llvm.intr.vector.reduce.fadd"(%arg0, %arg1) {reassoc = true} : (f32, vector<8xf32>) -> f32
  // CHECK: call reassoc float @llvm.vector.reduce.fmul.v8f32
  "llvm.intr.vector.reduce.fmul"(%arg0, %arg1) {reassoc = true} : (f32, vector<8xf32>) -> f32
  // CHECK: call i32 @llvm.vector.reduce.xor.v8i32
  "llvm.intr.vector.reduce.xor"(%arg2) : (vector<8xi32>) -> i32
  llvm.return
}

// CHECK-LABEL: @matrix_intrinsics
//                                       4x16                       16x3
llvm.func @matrix_intrinsics(%A: vector<64 x f32>, %B: vector<48 x f32>,
                             %ptr: !llvm.ptr<f32>, %stride: i64) {
  // CHECK: call <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float> %0, <48 x float> %1, i32 4, i32 16, i32 3)
  %C = llvm.intr.matrix.multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32} :
    (vector<64 x f32>, vector<48 x f32>) -> vector<12 x f32>
  // CHECK: call <48 x float> @llvm.matrix.transpose.v48f32(<48 x float> %1, i32 3, i32 16)
  %D = llvm.intr.matrix.transpose %B { rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> into vector<48 x f32>
  // CHECK: call <48 x float> @llvm.matrix.column.major.load.v48f32(float* align 4 %2, i64 %3, i1 false, i32 3, i32 16)
  %E = llvm.intr.matrix.column.major.load %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> from !llvm.ptr<f32> stride i64
  // CHECK: call void @llvm.matrix.column.major.store.v48f32(<48 x float> %7, float* align 4 %2, i64 %3, i1 false, i32 3, i32 16)
  llvm.intr.matrix.column.major.store %E, %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> to !llvm.ptr<f32> stride i64
  llvm.return
}

// CHECK-LABEL: @get_active_lane_mask
llvm.func @get_active_lane_mask(%base: i64, %n: i64) -> (vector<7xi1>) {
  // CHECK: call <7 x i1> @llvm.get.active.lane.mask.v7i1.i64(i64 %0, i64 %1)
  %0 = llvm.intr.get.active.lane.mask %base, %n : i64, i64 to vector<7xi1>
  llvm.return %0 : vector<7xi1>
}

// CHECK-LABEL: @masked_load_store_intrinsics
llvm.func @masked_load_store_intrinsics(%A: !llvm.ptr<vector<7xf32>>, %mask: vector<7xi1>) {
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>* %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> undef)
  %a = llvm.intr.masked.load %A, %mask { alignment = 1: i32} :
    (!llvm.ptr<vector<7xf32>>, vector<7xi1>) -> vector<7xf32>
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>* %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> %{{.*}})
  %b = llvm.intr.masked.load %A, %mask, %a { alignment = 1: i32} :
    (!llvm.ptr<vector<7xf32>>, vector<7xi1>, vector<7xf32>) -> vector<7xf32>
  // CHECK: call void @llvm.masked.store.v7f32.p0v7f32(<7 x float> %{{.*}}, <7 x float>* %0, i32 {{.*}}, <7 x i1> %{{.*}})
  llvm.intr.masked.store %b, %A, %mask { alignment = 1: i32} :
    vector<7xf32>, vector<7xi1> into !llvm.ptr<vector<7xf32>>
  llvm.return
}

// CHECK-LABEL: @masked_gather_scatter_intrinsics
llvm.func @masked_gather_scatter_intrinsics(%M: !llvm.vec<7 x ptr<f32>>, %mask: vector<7xi1>) {
  // CHECK: call <7 x float> @llvm.masked.gather.v7f32.v7p0f32(<7 x float*> %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> undef)
  %a = llvm.intr.masked.gather %M, %mask { alignment = 1: i32} :
      (!llvm.vec<7 x ptr<f32>>, vector<7xi1>) -> vector<7xf32>
  // CHECK: call <7 x float> @llvm.masked.gather.v7f32.v7p0f32(<7 x float*> %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> %{{.*}})
  %b = llvm.intr.masked.gather %M, %mask, %a { alignment = 1: i32} :
      (!llvm.vec<7 x ptr<f32>>, vector<7xi1>, vector<7xf32>) -> vector<7xf32>
  // CHECK: call void @llvm.masked.scatter.v7f32.v7p0f32(<7 x float> %{{.*}}, <7 x float*> %{{.*}}, i32 1, <7 x i1> %{{.*}})
  llvm.intr.masked.scatter %b, %M, %mask { alignment = 1: i32} :
      vector<7xf32>, vector<7xi1> into !llvm.vec<7 x ptr<f32>>
  llvm.return
}

// CHECK-LABEL: @masked_expand_compress_intrinsics
llvm.func @masked_expand_compress_intrinsics(%ptr: !llvm.ptr<f32>, %mask: vector<7xi1>, %passthru: vector<7xf32>) {
  // CHECK: call <7 x float> @llvm.masked.expandload.v7f32(float* %{{.*}}, <7 x i1> %{{.*}}, <7 x float> %{{.*}})
  %0 = "llvm.intr.masked.expandload"(%ptr, %mask, %passthru)
    : (!llvm.ptr<f32>, vector<7xi1>, vector<7xf32>) -> (vector<7xf32>)
  // CHECK: call void @llvm.masked.compressstore.v7f32(<7 x float> %{{.*}}, float* %{{.*}}, <7 x i1> %{{.*}})
  "llvm.intr.masked.compressstore"(%0, %ptr, %mask)
    : (vector<7xf32>, !llvm.ptr<f32>, vector<7xi1>) -> ()
  llvm.return
}

// CHECK-LABEL: @memcpy_test
llvm.func @memcpy_test(%arg0: i32, %arg2: !llvm.ptr<i8>, %arg3: !llvm.ptr<i8>) {
  %i1 = llvm.mlir.constant(false) : i1
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %{{.*}}, i8* %{{.*}}, i32 %{{.*}}, i1 {{.*}})
  "llvm.intr.memcpy"(%arg2, %arg3, %arg0, %i1) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()
  %sz = llvm.mlir.constant(10: i64) : i64
  // CHECK: call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* %{{.*}}, i8* %{{.*}}, i64 10, i1 {{.*}})
  "llvm.intr.memcpy.inline"(%arg2, %arg3, %sz, %i1) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  llvm.return
}

// CHECK-LABEL: @sadd_with_overflow_test
llvm.func @sadd_with_overflow_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32
  "llvm.intr.sadd.with.overflow"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: call { <8 x i32>, <8 x i1> } @llvm.sadd.with.overflow.v8i32
  "llvm.intr.sadd.with.overflow"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  llvm.return
}

// CHECK-LABEL: @uadd_with_overflow_test
llvm.func @uadd_with_overflow_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call { i32, i1 } @llvm.uadd.with.overflow.i32
  "llvm.intr.uadd.with.overflow"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: call { <8 x i32>, <8 x i1> } @llvm.uadd.with.overflow.v8i32
  "llvm.intr.uadd.with.overflow"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  llvm.return
}

// CHECK-LABEL: @ssub_with_overflow_test
llvm.func @ssub_with_overflow_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call { i32, i1 } @llvm.ssub.with.overflow.i32
  "llvm.intr.ssub.with.overflow"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: call { <8 x i32>, <8 x i1> } @llvm.ssub.with.overflow.v8i32
  "llvm.intr.ssub.with.overflow"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  llvm.return
}

// CHECK-LABEL: @usub_with_overflow_test
llvm.func @usub_with_overflow_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call { i32, i1 } @llvm.usub.with.overflow.i32
  "llvm.intr.usub.with.overflow"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: call { <8 x i32>, <8 x i1> } @llvm.usub.with.overflow.v8i32
  "llvm.intr.usub.with.overflow"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  llvm.return
}

// CHECK-LABEL: @smul_with_overflow_test
llvm.func @smul_with_overflow_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call { i32, i1 } @llvm.smul.with.overflow.i32
  "llvm.intr.smul.with.overflow"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: call { <8 x i32>, <8 x i1> } @llvm.smul.with.overflow.v8i32
  "llvm.intr.smul.with.overflow"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  llvm.return
}

// CHECK-LABEL: @umul_with_overflow_test
llvm.func @umul_with_overflow_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call { i32, i1 } @llvm.umul.with.overflow.i32
  "llvm.intr.umul.with.overflow"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: call { <8 x i32>, <8 x i1> } @llvm.umul.with.overflow.v8i32
  "llvm.intr.umul.with.overflow"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> !llvm.struct<(vector<8xi32>, vector<8xi1>)>
  llvm.return
}

// CHECK-LABEL: @coro_id
llvm.func @coro_id(%arg0: i32, %arg1: !llvm.ptr<i8>) {
  // CHECK: call token @llvm.coro.id
  %null = llvm.mlir.null : !llvm.ptr<i8>
  llvm.intr.coro.id %arg0, %arg1, %arg1, %null : !llvm.token
  llvm.return
}

// CHECK-LABEL: @coro_begin
llvm.func @coro_begin(%arg0: i32, %arg1: !llvm.ptr<i8>) {
  %null = llvm.mlir.null : !llvm.ptr<i8>
  %token = llvm.intr.coro.id %arg0, %arg1, %arg1, %null : !llvm.token
  // CHECK: call i8* @llvm.coro.begin
  llvm.intr.coro.begin %token, %arg1 : !llvm.ptr<i8>
  llvm.return
}

// CHECK-LABEL: @coro_size
llvm.func @coro_size() {
  // CHECK: call i64 @llvm.coro.size.i64
  %0 = llvm.intr.coro.size : i64
  // CHECK: call i32 @llvm.coro.size.i32
  %1 = llvm.intr.coro.size : i32
  llvm.return
}

// CHECK-LABEL: @coro_save
llvm.func @coro_save(%arg0: !llvm.ptr<i8>) {
  // CHECK: call token @llvm.coro.save
  %0 = llvm.intr.coro.save %arg0 : !llvm.token
  llvm.return
}

// CHECK-LABEL: @coro_suspend
llvm.func @coro_suspend(%arg0: i32, %arg1 : i1, %arg2 : !llvm.ptr<i8>) {
  %null = llvm.mlir.null : !llvm.ptr<i8>
  %token = llvm.intr.coro.id %arg0, %arg2, %arg2, %null : !llvm.token
  // CHECK: call i8 @llvm.coro.suspend
  %0 = llvm.intr.coro.suspend %token, %arg1 : i8
  llvm.return
}

// CHECK-LABEL: @coro_end
llvm.func @coro_end(%arg0: !llvm.ptr<i8>, %arg1 : i1) {
  // CHECK: call i1 @llvm.coro.end
  %0 = llvm.intr.coro.end %arg0, %arg1 : i1
  llvm.return
}

// CHECK-LABEL: @coro_free
llvm.func @coro_free(%arg0: i32, %arg1 : !llvm.ptr<i8>) {
  %null = llvm.mlir.null : !llvm.ptr<i8>
  %token = llvm.intr.coro.id %arg0, %arg1, %arg1, %null : !llvm.token
  // CHECK: call i8* @llvm.coro.free
  %0 = llvm.intr.coro.free %token, %arg1 : !llvm.ptr<i8>
  llvm.return
}

// CHECK-LABEL: @coro_resume
llvm.func @coro_resume(%arg0: !llvm.ptr<i8>) {
  // CHECK: call void @llvm.coro.resume
  llvm.intr.coro.resume %arg0
  llvm.return
}

// CHECK-LABEL: @stack_save
llvm.func @stack_save() {
  // CHECK: call i8* @llvm.stacksave
  %0 = llvm.intr.stacksave : !llvm.ptr<i8>
  llvm.return
}

// CHECK-LABEL: @stack_restore
llvm.func @stack_restore(%arg0: !llvm.ptr<i8>) {
  // CHECK: call void @llvm.stackrestore
  llvm.intr.stackrestore %arg0
  llvm.return
}

// Check that intrinsics are declared with appropriate types.
// CHECK-DAG: declare float @llvm.fma.f32(float, float, float)
// CHECK-DAG: declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
// CHECK-DAG: declare float @llvm.fmuladd.f32(float, float, float)
// CHECK-DAG: declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
// CHECK-DAG: declare void @llvm.prefetch.p0i8(i8* nocapture readonly, i32 immarg, i32 immarg, i32)
// CHECK-DAG: declare float @llvm.exp.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.exp.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.log.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.log.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.log10.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.log10.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.log2.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.log2.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.fabs.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.sqrt.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.ceil.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.ceil.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.cos.f32(float)
// CHECK-DAG: declare <8 x float> @llvm.cos.v8f32(<8 x float>) #0
// CHECK-DAG: declare float @llvm.copysign.f32(float, float)
// CHECK-DAG: declare <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float>, <48 x float>, i32 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare <48 x float> @llvm.matrix.transpose.v48f32(<48 x float>, i32 immarg, i32 immarg)
// CHECK-DAG: declare <48 x float> @llvm.matrix.column.major.load.v48f32(float* nocapture, i64, i1 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare void @llvm.matrix.column.major.store.v48f32(<48 x float>, float* nocapture writeonly, i64, i1 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare <7 x i1> @llvm.get.active.lane.mask.v7i1.i64(i64, i64)
// CHECK-DAG: declare <7 x float> @llvm.masked.load.v7f32.p0v7f32(<7 x float>*, i32 immarg, <7 x i1>, <7 x float>)
// CHECK-DAG: declare void @llvm.masked.store.v7f32.p0v7f32(<7 x float>, <7 x float>*, i32 immarg, <7 x i1>)
// CHECK-DAG: declare <7 x float> @llvm.masked.gather.v7f32.v7p0f32(<7 x float*>, i32 immarg, <7 x i1>, <7 x float>)
// CHECK-DAG: declare void @llvm.masked.scatter.v7f32.v7p0f32(<7 x float>, <7 x float*>, i32 immarg, <7 x i1>)
// CHECK-DAG: declare <7 x float> @llvm.masked.expandload.v7f32(float*, <7 x i1>, <7 x float>)
// CHECK-DAG: declare void @llvm.masked.compressstore.v7f32(<7 x float>, float*, <7 x i1>)
// CHECK-DAG: declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg)
// CHECK-DAG: declare void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64 immarg, i1 immarg)
// CHECK-DAG: declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.sadd.with.overflow.v8i32(<8 x i32>, <8 x i32>) #0
// CHECK-DAG: declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.uadd.with.overflow.v8i32(<8 x i32>, <8 x i32>) #0
// CHECK-DAG: declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.ssub.with.overflow.v8i32(<8 x i32>, <8 x i32>) #0
// CHECK-DAG: declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.usub.with.overflow.v8i32(<8 x i32>, <8 x i32>) #0
// CHECK-DAG: declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.umul.with.overflow.v8i32(<8 x i32>, <8 x i32>) #0
// CHECK-DAG: declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
// CHECK-DAG: declare i8* @llvm.coro.begin(token, i8* writeonly)
// CHECK-DAG: declare i64 @llvm.coro.size.i64()
// CHECK-DAG: declare i32 @llvm.coro.size.i32()
// CHECK-DAG: declare token @llvm.coro.save(i8*)
// CHECK-DAG: declare i8 @llvm.coro.suspend(token, i1)
// CHECK-DAG: declare i1 @llvm.coro.end(i8*, i1)
// CHECK-DAG: declare i8* @llvm.coro.free(token, i8* nocapture readonly)
// CHECK-DAG: declare void @llvm.coro.resume(i8*)
