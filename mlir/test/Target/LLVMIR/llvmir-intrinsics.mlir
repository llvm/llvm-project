// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @intrinsics
llvm.func @intrinsics(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: !llvm.ptr) {
  // CHECK: call float @llvm.fmuladd.f32
  "llvm.intr.fmuladd"(%arg0, %arg1, %arg0) : (f32, f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.fmuladd.v8f32
  "llvm.intr.fmuladd"(%arg2, %arg2, %arg2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK: call float @llvm.fma.f32
  "llvm.intr.fma"(%arg0, %arg1, %arg0) : (f32, f32, f32) -> f32
  // CHECK: call <8 x float> @llvm.fma.v8f32
  "llvm.intr.fma"(%arg2, %arg2, %arg2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK: call void @llvm.prefetch.p0(ptr %3, i32 0, i32 3, i32 1)
  "llvm.intr.prefetch"(%arg3) <{cache = 1 : i32, hint = 3 : i32, rw = 0 : i32}> : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: @fpclass_test
llvm.func @fpclass_test(%arg0: f32) -> i1 {
  // CHECK: call i1 @llvm.is.fpclass
  %0 = "llvm.intr.is.fpclass"(%arg0) <{bit = 0 : i32 }>: (f32) -> i1
  llvm.return %0 : i1
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

// CHECK-LABEL: @hyperbolic_trig_test
llvm.func @hyperbolic_trig_test(%arg0: f32, %arg1: vector<8xf32>) {
  // CHECK: call float @llvm.sinh.f32
  llvm.intr.sinh(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.sinh.v8f32
  llvm.intr.sinh(%arg1) : (vector<8xf32>) -> vector<8xf32>

  // CHECK: call float @llvm.cosh.f32
  llvm.intr.cosh(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.cosh.v8f32
  llvm.intr.cosh(%arg1) : (vector<8xf32>) -> vector<8xf32>

  // CHECK: call float @llvm.tanh.f32
  llvm.intr.tanh(%arg0) : (f32) -> f32
  // CHECK: call <8 x float> @llvm.tanh.v8f32
  llvm.intr.tanh(%arg1) : (vector<8xf32>) -> vector<8xf32>
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

// CHECK-LABEL: @rint_test
llvm.func @rint_test(%arg0 : f32, %arg1 : f64, %arg2 : vector<8xf32>, %arg3 : vector<8xf64>) {
  // CHECK: call float @llvm.rint.f32
  "llvm.intr.rint"(%arg0) : (f32) -> f32
  // CHECK: call double @llvm.rint.f64
  "llvm.intr.rint"(%arg1) : (f64) -> f64
  // CHECK: call <8 x float> @llvm.rint.v8f32
  "llvm.intr.rint"(%arg2) : (vector<8xf32>) -> vector<8xf32>
  // CHECK: call <8 x double> @llvm.rint.v8f64
  "llvm.intr.rint"(%arg3) : (vector<8xf64>) -> vector<8xf64>
  llvm.return
}

// CHECK-LABEL: @nearbyint_test
llvm.func @nearbyint_test(%arg0 : f32, %arg1 : f64, %arg2 : vector<8xf32>, %arg3 : vector<8xf64>) {
  // CHECK: call float @llvm.nearbyint.f32
  "llvm.intr.nearbyint"(%arg0) : (f32) -> f32
  // CHECK: call double @llvm.nearbyint.f64
  "llvm.intr.nearbyint"(%arg1) : (f64) -> f64
  // CHECK: call <8 x float> @llvm.nearbyint.v8f32
  "llvm.intr.nearbyint"(%arg2) : (vector<8xf32>) -> vector<8xf32>
  // CHECK: call <8 x double> @llvm.nearbyint.v8f64
  "llvm.intr.nearbyint"(%arg3) : (vector<8xf64>) -> vector<8xf64>
  llvm.return
}

// CHECK-LABEL: @lround_test
llvm.func @lround_test(%arg0 : f32, %arg1 : f64) {
  // CHECK: call i32 @llvm.lround.i32.f32
  "llvm.intr.lround"(%arg0) : (f32) -> i32
  // CHECK: call i64 @llvm.lround.i64.f32
  "llvm.intr.lround"(%arg0) : (f32) -> i64
  // CHECK: call i32 @llvm.lround.i32.f64
  "llvm.intr.lround"(%arg1) : (f64) -> i32
  // CHECK: call i64 @llvm.lround.i64.f64
  "llvm.intr.lround"(%arg1) : (f64) -> i64
  llvm.return
}

// CHECK-LABEL: @llround_test
llvm.func @llround_test(%arg0 : f32, %arg1 : f64) {
  // CHECK: call i64 @llvm.llround.i64.f32
  "llvm.intr.llround"(%arg0) : (f32) -> i64
  // CHECK: call i64 @llvm.llround.i64.f64
  "llvm.intr.llround"(%arg1) : (f64) -> i64
  llvm.return
}

// CHECK-LABEL: @lrint_test
llvm.func @lrint_test(%arg0 : f32, %arg1 : f64) {
  // CHECK: call i32 @llvm.lrint.i32.f32
  "llvm.intr.lrint"(%arg0) : (f32) -> i32
  // CHECK: call i64 @llvm.lrint.i64.f32
  "llvm.intr.lrint"(%arg0) : (f32) -> i64
  // CHECK: call i32 @llvm.lrint.i32.f64
  "llvm.intr.lrint"(%arg1) : (f64) -> i32
  // CHECK: call i64 @llvm.lrint.i64.f64
  "llvm.intr.lrint"(%arg1) : (f64) -> i64
  llvm.return
}

// CHECK-LABEL: @llrint_test
llvm.func @llrint_test(%arg0 : f32, %arg1 : f64) {
  // CHECK: call i64 @llvm.llrint.i64.f32
  "llvm.intr.llrint"(%arg0) : (f32) -> i64
  // CHECK: call i64 @llvm.llrint.i64.f64
  "llvm.intr.llrint"(%arg1) : (f64) -> i64
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

// CHECK-LABEL: @byteswap_test
llvm.func @byteswap_test(%arg0: i32, %arg1: vector<8xi32>) {
  // CHECK: call i32 @llvm.bswap.i32
  "llvm.intr.bswap"(%arg0) : (i32) -> i32
  // CHECK: call <8 x i32> @llvm.bswap.v8i32
  "llvm.intr.bswap"(%arg1) : (vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @ctlz_test
llvm.func @ctlz_test(%arg0: i32, %arg1: vector<8xi32>) {
  // CHECK: call i32 @llvm.ctlz.i32
  "llvm.intr.ctlz"(%arg0) <{is_zero_poison = 0 : i1}> : (i32) -> i32
  // CHECK: call <8 x i32> @llvm.ctlz.v8i32
  "llvm.intr.ctlz"(%arg1) <{is_zero_poison = 1 : i1}> : (vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @cttz_test
llvm.func @cttz_test(%arg0: i32, %arg1: vector<8xi32>) {
  // CHECK: call i32 @llvm.cttz.i32
  "llvm.intr.cttz"(%arg0) <{is_zero_poison = 0 : i1}> : (i32) -> i32
  // CHECK: call <8 x i32> @llvm.cttz.v8i32
  "llvm.intr.cttz"(%arg1) <{is_zero_poison = 1 : i1}> : (vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @abs_test
llvm.func @abs_test(%arg0: i32, %arg1: vector<8xi32>) {
  // CHECK: call i32 @llvm.abs.i32
  "llvm.intr.abs"(%arg0) <{is_int_min_poison = 0 : i1}> : (i32) -> i32
  // CHECK: call <8 x i32> @llvm.abs.v8i32
  "llvm.intr.abs"(%arg1) <{is_int_min_poison = 1 : i1}> : (vector<8xi32>) -> vector<8xi32>
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

// CHECK-LABEL: @fshl_test
llvm.func @fshl_test(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: vector<8xi32>, %arg4: vector<8xi32>, %arg5: vector<8xi32>) {
  // CHECK: call i32 @llvm.fshl.i32
  "llvm.intr.fshl"(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32
  "llvm.intr.fshl"(%arg3, %arg4, %arg5) : (vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @fshr_test
llvm.func @fshr_test(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: vector<8xi32>, %arg4: vector<8xi32>, %arg5: vector<8xi32>) {
  // CHECK: call i32 @llvm.fshr.i32
  "llvm.intr.fshr"(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32
  "llvm.intr.fshr"(%arg3, %arg4, %arg5) : (vector<8xi32>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
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

// CHECK-LABEL: @umax_test
llvm.func @umax_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.umax.i32
  "llvm.intr.umax"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.umax.v8i32
  "llvm.intr.umax"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @umin_test
llvm.func @umin_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.umin.i32
  "llvm.intr.umin"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.umin.v8i32
  "llvm.intr.umin"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @assume_without_opbundles
llvm.func @assume_without_opbundles(%cond: i1) {
  // CHECK: call void @llvm.assume(i1 %{{.+}})
  llvm.intr.assume %cond : i1
  llvm.return
}

// CHECK-LABEL: @assume_with_opbundles
llvm.func @assume_with_opbundles(%cond: i1, %p: !llvm.ptr) {
  %0 = llvm.mlir.constant(8 : i32) : i32
  // CHECK: call void @llvm.assume(i1 %{{.+}}) [ "align"(ptr %{{.+}}, i32 8) ]
  llvm.intr.assume %cond ["align"(%p, %0 : !llvm.ptr, i32)] : i1
  llvm.return
}

// CHECK-LABEL: @vector_reductions
llvm.func @vector_reductions(%arg0: f32, %arg1: vector<8xf32>, %arg2: vector<8xi32>) {
  // CHECK: call i32 @llvm.vector.reduce.add.v8i32
  "llvm.intr.vector.reduce.add"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call i32 @llvm.vector.reduce.and.v8i32
  "llvm.intr.vector.reduce.and"(%arg2) : (vector<8xi32>) -> i32
  // CHECK: call float @llvm.vector.reduce.fmax.v8f32
  llvm.intr.vector.reduce.fmax(%arg1) : (vector<8xf32>) -> f32
  // CHECK: call float @llvm.vector.reduce.fmin.v8f32
  llvm.intr.vector.reduce.fmin(%arg1) : (vector<8xf32>) -> f32
  // CHECK: call float @llvm.vector.reduce.fmaximum.v8f32
  llvm.intr.vector.reduce.fmaximum(%arg1) : (vector<8xf32>) -> f32
  // CHECK: call float @llvm.vector.reduce.fminimum.v8f32
  llvm.intr.vector.reduce.fminimum(%arg1) : (vector<8xf32>) -> f32
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
  "llvm.intr.vector.reduce.fadd"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f32, vector<8xf32>) -> f32
  // CHECK: call reassoc float @llvm.vector.reduce.fmul.v8f32
  "llvm.intr.vector.reduce.fmul"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f32, vector<8xf32>) -> f32
  // CHECK: call i32 @llvm.vector.reduce.xor.v8i32
  "llvm.intr.vector.reduce.xor"(%arg2) : (vector<8xi32>) -> i32
  llvm.return
}

// CHECK-LABEL: @matrix_intrinsics
//                                       4x16                       16x3
llvm.func @matrix_intrinsics(%A: vector<64 x f32>, %B: vector<48 x f32>,
                             %ptr: !llvm.ptr, %stride: i64) {
  // CHECK: call <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float> %0, <48 x float> %1, i32 4, i32 16, i32 3)
  %C = llvm.intr.matrix.multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32} :
    (vector<64 x f32>, vector<48 x f32>) -> vector<12 x f32>
  // CHECK: call <48 x float> @llvm.matrix.transpose.v48f32(<48 x float> %1, i32 3, i32 16)
  %D = llvm.intr.matrix.transpose %B { rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> into vector<48 x f32>
  // CHECK: call <48 x float> @llvm.matrix.column.major.load.v48f32.i64(ptr align 4 %2, i64 %3, i1 false, i32 3, i32 16)
  %E = llvm.intr.matrix.column.major.load %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> from !llvm.ptr stride i64
  // CHECK: call void @llvm.matrix.column.major.store.v48f32.i64(<48 x float> %7, ptr align 4 %2, i64 %3, i1 false, i32 3, i32 16)
  llvm.intr.matrix.column.major.store %E, %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} :
    vector<48 x f32> to !llvm.ptr stride i64
  llvm.return
}

// CHECK-LABEL: @get_active_lane_mask
llvm.func @get_active_lane_mask(%base: i64, %n: i64) -> (vector<7xi1>) {
  // CHECK: call <7 x i1> @llvm.get.active.lane.mask.v7i1.i64(i64 %0, i64 %1)
  %0 = llvm.intr.get.active.lane.mask %base, %n : i64, i64 to vector<7xi1>
  llvm.return %0 : vector<7xi1>
}

// CHECK-LABEL: @masked_load_store_intrinsics
llvm.func @masked_load_store_intrinsics(%A: !llvm.ptr, %mask: vector<7xi1>) {
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0(ptr %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> poison)
  %a = llvm.intr.masked.load %A, %mask { alignment = 1: i32} :
    (!llvm.ptr, vector<7xi1>) -> vector<7xf32>
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0(ptr %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> poison), !nontemporal !1
  %b = llvm.intr.masked.load %A, %mask { alignment = 1: i32, nontemporal} :
    (!llvm.ptr, vector<7xi1>) -> vector<7xf32>
  // CHECK: call <7 x float> @llvm.masked.load.v7f32.p0(ptr %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> %{{.*}})
  %c = llvm.intr.masked.load %A, %mask, %a { alignment = 1: i32} :
    (!llvm.ptr, vector<7xi1>, vector<7xf32>) -> vector<7xf32>
  // CHECK: call void @llvm.masked.store.v7f32.p0(<7 x float> %{{.*}}, ptr %0, i32 {{.*}}, <7 x i1> %{{.*}})
  llvm.intr.masked.store %b, %A, %mask { alignment = 1: i32} :
    vector<7xf32>, vector<7xi1> into !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @masked_gather_scatter_intrinsics
llvm.func @masked_gather_scatter_intrinsics(%M: !llvm.vec<7 x ptr>, %mask: vector<7xi1>) {
  // CHECK: call <7 x float> @llvm.masked.gather.v7f32.v7p0(<7 x ptr> %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> poison)
  %a = llvm.intr.masked.gather %M, %mask { alignment = 1: i32} :
      (!llvm.vec<7 x ptr>, vector<7xi1>) -> vector<7xf32>
  // CHECK: call <7 x float> @llvm.masked.gather.v7f32.v7p0(<7 x ptr> %{{.*}}, i32 1, <7 x i1> %{{.*}}, <7 x float> %{{.*}})
  %b = llvm.intr.masked.gather %M, %mask, %a { alignment = 1: i32} :
      (!llvm.vec<7 x ptr>, vector<7xi1>, vector<7xf32>) -> vector<7xf32>
  // CHECK: call void @llvm.masked.scatter.v7f32.v7p0(<7 x float> %{{.*}}, <7 x ptr> %{{.*}}, i32 1, <7 x i1> %{{.*}})
  llvm.intr.masked.scatter %b, %M, %mask { alignment = 1: i32} :
      vector<7xf32>, vector<7xi1> into !llvm.vec<7 x ptr>
  llvm.return
}

// CHECK-LABEL: @masked_expand_compress_intrinsics
llvm.func @masked_expand_compress_intrinsics(%ptr: !llvm.ptr, %mask: vector<7xi1>, %passthru: vector<7xf32>) {
  // CHECK: call <7 x float> @llvm.masked.expandload.v7f32(ptr %{{.*}}, <7 x i1> %{{.*}}, <7 x float> %{{.*}})
  %0 = "llvm.intr.masked.expandload"(%ptr, %mask, %passthru)
    : (!llvm.ptr, vector<7xi1>, vector<7xf32>) -> (vector<7xf32>)
  // CHECK: call void @llvm.masked.compressstore.v7f32(<7 x float> %{{.*}}, ptr %{{.*}}, <7 x i1> %{{.*}})
  "llvm.intr.masked.compressstore"(%0, %ptr, %mask)
    : (vector<7xf32>, !llvm.ptr, vector<7xi1>) -> ()
  llvm.return
}

// CHECK-LABEL: @annotate_intrinsics
llvm.func @annotate_intrinsics(%var: !llvm.ptr, %int: i16, %ptr: !llvm.ptr, %annotation: !llvm.ptr, %fileName: !llvm.ptr, %line: i32, %attr: !llvm.ptr) {
  // CHECK: call void @llvm.var.annotation.p0.p0(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  "llvm.intr.var.annotation"(%var, %annotation, %fileName, %line, %attr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  // CHECK: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  %res0 = "llvm.intr.ptr.annotation"(%ptr, %annotation, %fileName, %line, %attr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> (!llvm.ptr)
  // CHECK: call i16 @llvm.annotation.i16.p0(i16 %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}})
  %res1 = "llvm.intr.annotation"(%int, %annotation, %fileName, %line) : (i16, !llvm.ptr, !llvm.ptr, i32) -> (i16)
  llvm.return
}

// CHECK-LABEL: @trap_intrinsics
llvm.func @trap_intrinsics() {
  // CHECK: call void @llvm.trap()
  "llvm.intr.trap"() : () -> ()
  // CHECK: call void @llvm.debugtrap()
  "llvm.intr.debugtrap"() : () -> ()
  // CHECK: call void @llvm.ubsantrap(i8 1)
  "llvm.intr.ubsantrap"() {failureKind = 1 : i8} : () -> ()
  llvm.return
}

// CHECK-LABEL: @memcpy_test
llvm.func @memcpy_test(%arg0: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}}, i1 false
  "llvm.intr.memcpy"(%arg2, %arg3, %arg0) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  // CHECK: call void @llvm.memcpy.inline.p0.p0.i32(ptr %{{.*}}, ptr %{{.*}}, i32 10, i1 true
  "llvm.intr.memcpy.inline"(%arg2, %arg3) <{isVolatile = true, len = 10 : i32}> : (!llvm.ptr, !llvm.ptr) -> ()
  // CHECK: call void @llvm.memcpy.inline.p0.p0.i64(ptr %{{.*}}, ptr %{{.*}}, i64 10, i1 true
  "llvm.intr.memcpy.inline"(%arg2, %arg3) <{isVolatile = true, len = 10 : i64}> : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: @memmove_test
llvm.func @memmove_test(%arg0: i32, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
  // CHECK: call void @llvm.memmove.p0.p0.i32(ptr %{{.*}}, ptr %{{.*}}, i32 %{{.*}}, i1 false
  "llvm.intr.memmove"(%arg2, %arg3, %arg0) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  llvm.return
}

// CHECK-LABEL: @memset_test
llvm.func @memset_test(%arg0: i32, %arg2: !llvm.ptr, %arg3: i8) {
  %i1 = llvm.mlir.constant(false) : i1
  // CHECK: call void @llvm.memset.p0.i32(ptr %{{.*}}, i8 %{{.*}}, i32 %{{.*}}, i1 false
  "llvm.intr.memset"(%arg2, %arg3, %arg0) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
  // CHECK: call void @llvm.memset.inline.p0.i32(ptr %{{.*}}, i8 %{{.*}}, i32 10, i1 true
  "llvm.intr.memset.inline"(%arg2, %arg3) <{isVolatile = true, len = 10 : i32}> : (!llvm.ptr, i8) -> ()
  // CHECK: call void @llvm.memset.inline.p0.i64(ptr %{{.*}}, i8 %{{.*}}, i64 10, i1 true
  "llvm.intr.memset.inline"(%arg2, %arg3) <{isVolatile = true, len = 10 : i64}> : (!llvm.ptr, i8) -> ()
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

// CHECK-LABEL: @is_constant
llvm.func @is_constant(%arg0: i32) {
  // CHECK: call i1 @llvm.is.constant.i32(i32 %{{.*}})
  %0 = "llvm.intr.is.constant"(%arg0) : (i32) -> i1
  llvm.return
}

// CHECK-LABEL: @expect
llvm.func @expect(%arg0: i32) {
  %0 = llvm.mlir.constant(42 : i32) : i32
  // CHECK: call i32 @llvm.expect.i32(i32 %{{.*}}, i32 42)
  %1 = llvm.intr.expect %arg0, %0 : i32
  llvm.return
}

// CHECK-LABEL: @expect_with_probability
llvm.func @expect_with_probability(%arg0: i16) {
  %0 = llvm.mlir.constant(42 : i16) : i16
  // CHECK: call i16 @llvm.expect.with.probability.i16(i16 %{{.*}}, i16 42, double 5.000000e-01)
  %1 = llvm.intr.expect.with.probability %arg0, %0, 5.000000e-01 : i16
  llvm.return
}

llvm.mlir.global external thread_local @tls_var(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32

// CHECK-LABEL: @threadlocal_test
llvm.func @threadlocal_test() {
  // CHECK: call ptr @llvm.threadlocal.address.p0(ptr @tls_var)
  %0 = llvm.mlir.addressof @tls_var : !llvm.ptr
  "llvm.intr.threadlocal.address"(%0) : (!llvm.ptr) -> !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @sadd_sat_test
llvm.func @sadd_sat_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.sadd.sat.i32
  "llvm.intr.sadd.sat"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.sadd.sat.v8i32
  "llvm.intr.sadd.sat"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @uadd_sat_test
llvm.func @uadd_sat_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.uadd.sat.i32
  "llvm.intr.uadd.sat"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.uadd.sat.v8i32
  "llvm.intr.uadd.sat"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @ssub_sat_test
llvm.func @ssub_sat_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.ssub.sat.i32
  "llvm.intr.ssub.sat"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.ssub.sat.v8i32
  "llvm.intr.ssub.sat"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @usub_sat_test
llvm.func @usub_sat_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.usub.sat.i32
  "llvm.intr.usub.sat"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.usub.sat.v8i32
  "llvm.intr.usub.sat"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @sshl_sat_test
llvm.func @sshl_sat_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.sshl.sat.i32
  "llvm.intr.sshl.sat"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.sshl.sat.v8i32
  "llvm.intr.sshl.sat"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @ushl_sat_test
llvm.func @ushl_sat_test(%arg0: i32, %arg1: i32, %arg2: vector<8xi32>, %arg3: vector<8xi32>) {
  // CHECK: call i32 @llvm.ushl.sat.i32
  "llvm.intr.ushl.sat"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK: call <8 x i32> @llvm.ushl.sat.v8i32
  "llvm.intr.ushl.sat"(%arg2, %arg3) : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @coro_id
llvm.func @coro_id(%arg0: i32, %arg1: !llvm.ptr) {
  // CHECK: call token @llvm.coro.id
  %null = llvm.mlir.zero : !llvm.ptr
  llvm.intr.coro.id %arg0, %arg1, %arg1, %null : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
  llvm.return
}

// CHECK-LABEL: @coro_begin
llvm.func @coro_begin(%arg0: i32, %arg1: !llvm.ptr) {
  %null = llvm.mlir.zero : !llvm.ptr
  %token = llvm.intr.coro.id %arg0, %arg1, %arg1, %null : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
  // CHECK: call ptr @llvm.coro.begin
  llvm.intr.coro.begin %token, %arg1 : (!llvm.token, !llvm.ptr) -> !llvm.ptr
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

// CHECK-LABEL: @coro_align
llvm.func @coro_align() {
  // CHECK: call i64 @llvm.coro.align.i64
  %0 = llvm.intr.coro.align : i64
  // CHECK: call i32 @llvm.coro.align.i32
  %1 = llvm.intr.coro.align : i32
  llvm.return
}

// CHECK-LABEL: @coro_save
llvm.func @coro_save(%arg0: !llvm.ptr) {
  // CHECK: call token @llvm.coro.save
  %0 = llvm.intr.coro.save %arg0 : (!llvm.ptr) -> !llvm.token
  llvm.return
}

// CHECK-LABEL: @coro_suspend
llvm.func @coro_suspend(%arg0: i32, %arg1 : i1, %arg2 : !llvm.ptr) {
  %null = llvm.mlir.zero : !llvm.ptr
  %token = llvm.intr.coro.id %arg0, %arg2, %arg2, %null : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
  // CHECK: call i8 @llvm.coro.suspend
  %0 = llvm.intr.coro.suspend %token, %arg1 : i8
  llvm.return
}

// CHECK-LABEL: @coro_end
llvm.func @coro_end(%arg0: !llvm.ptr, %arg1 : i1) {
  %none = llvm.mlir.none : !llvm.token
  // CHECK: call i1 @llvm.coro.end
  %0 = llvm.intr.coro.end %arg0, %arg1, %none : (!llvm.ptr, i1, !llvm.token) -> i1
  llvm.return
}

// CHECK-LABEL: @coro_free
llvm.func @coro_free(%arg0: i32, %arg1 : !llvm.ptr) {
  %null = llvm.mlir.zero : !llvm.ptr
  %token = llvm.intr.coro.id %arg0, %arg1, %arg1, %null : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
  // CHECK: call ptr @llvm.coro.free
  %0 = llvm.intr.coro.free %token, %arg1 : (!llvm.token, !llvm.ptr) -> !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @coro_resume
llvm.func @coro_resume(%arg0: !llvm.ptr) {
  // CHECK: call void @llvm.coro.resume
  llvm.intr.coro.resume %arg0 : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @coro_promise
llvm.func @coro_promise(%arg0: !llvm.ptr, %arg1 : i32, %arg2 : i1) {
  // CHECK: call ptr @llvm.coro.promise
  %0 = llvm.intr.coro.promise %arg0, %arg1, %arg2 : (!llvm.ptr, i32, i1) -> !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @eh_typeid_for
llvm.func @eh_typeid_for(%arg0 : !llvm.ptr) {
    // CHECK: call i32 @llvm.eh.typeid.for.p0
    %0 = llvm.intr.eh.typeid.for %arg0 : (!llvm.ptr) -> i32
    llvm.return
}

// CHECK-LABEL: @stack_save
llvm.func @stack_save() {
  // CHECK: call ptr @llvm.stacksave.p0
  %0 = llvm.intr.stacksave : !llvm.ptr
  // CHECK: call ptr addrspace(1) @llvm.stacksave.p1
  %1 = llvm.intr.stacksave : !llvm.ptr<1>
  llvm.return
}

// CHECK-LABEL: @stack_restore
llvm.func @stack_restore(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>) {
  // CHECK: call void @llvm.stackrestore.p0(ptr %{{.}})
  llvm.intr.stackrestore %arg0 : !llvm.ptr
  // CHECK: call void @llvm.stackrestore.p1(ptr addrspace(1) %{{.}})
  llvm.intr.stackrestore %arg1 : !llvm.ptr<1>
  llvm.return
}

// CHECK-LABEL: @vector_predication_intrinsics
llvm.func @vector_predication_intrinsics(%A: vector<8xi32>, %B: vector<8xi32>,
                                         %C: vector<8xf32>, %D: vector<8xf32>,
                                         %E: vector<8xi64>, %F: vector<8xf64>,
                                         %G: !llvm.vec<8 x !llvm.ptr>,
                                         %i: i32, %f: f32,
                                         %iptr : !llvm.ptr,
                                         %fptr : !llvm.ptr,
                                         %mask: vector<8xi1>, %evl: i32) {
  // CHECK: call <8 x i32> @llvm.vp.add.v8i32
  "llvm.intr.vp.add" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.sub.v8i32
  "llvm.intr.vp.sub" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.mul.v8i32
  "llvm.intr.vp.mul" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.sdiv.v8i32
  "llvm.intr.vp.sdiv" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.udiv.v8i32
  "llvm.intr.vp.udiv" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.srem.v8i32
  "llvm.intr.vp.srem" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.urem.v8i32
  "llvm.intr.vp.urem" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.ashr.v8i32
  "llvm.intr.vp.ashr" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.lshr.v8i32
  "llvm.intr.vp.lshr" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.shl.v8i32
  "llvm.intr.vp.shl" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.or.v8i32
  "llvm.intr.vp.or" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.and.v8i32
  "llvm.intr.vp.and" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.xor.v8i32
  "llvm.intr.vp.xor" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.smax.v8i32
  "llvm.intr.vp.smax" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.smin.v8i32
  "llvm.intr.vp.smin" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.umax.v8i32
  "llvm.intr.vp.umax" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.umin.v8i32
  "llvm.intr.vp.umin" (%A, %B, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>

  // CHECK: call <8 x float> @llvm.vp.fadd.v8f32
  "llvm.intr.vp.fadd" (%C, %D, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x float> @llvm.vp.fsub.v8f32
  "llvm.intr.vp.fsub" (%C, %D, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x float> @llvm.vp.fmul.v8f32
  "llvm.intr.vp.fmul" (%C, %D, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x float> @llvm.vp.fdiv.v8f32
  "llvm.intr.vp.fdiv" (%C, %D, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x float> @llvm.vp.frem.v8f32
  "llvm.intr.vp.frem" (%C, %D, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x float> @llvm.vp.fneg.v8f32
  "llvm.intr.vp.fneg" (%C, %mask, %evl) :
         (vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x float> @llvm.vp.fma.v8f32
  "llvm.intr.vp.fma" (%C, %D, %D, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x float> @llvm.vp.fmuladd.v8f32
  "llvm.intr.vp.fmuladd" (%C, %D, %D, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>

  // CHECK: call i32 @llvm.vp.reduce.add.v8i32
  "llvm.intr.vp.reduce.add" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.mul.v8i32
  "llvm.intr.vp.reduce.mul" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.and.v8i32
  "llvm.intr.vp.reduce.and" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.or.v8i32
  "llvm.intr.vp.reduce.or" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.xor.v8i32
  "llvm.intr.vp.reduce.xor" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.smax.v8i32
  "llvm.intr.vp.reduce.smax" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.smin.v8i32
  "llvm.intr.vp.reduce.smin" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.umax.v8i32
  "llvm.intr.vp.reduce.umax" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  // CHECK: call i32 @llvm.vp.reduce.umin.v8i32
  "llvm.intr.vp.reduce.umin" (%i, %A, %mask, %evl) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32

  // CHECK: call float @llvm.vp.reduce.fadd.v8f32
  "llvm.intr.vp.reduce.fadd" (%f, %C, %mask, %evl) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  // CHECK: call float @llvm.vp.reduce.fmul.v8f32
  "llvm.intr.vp.reduce.fmul" (%f, %C, %mask, %evl) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  // CHECK: call float @llvm.vp.reduce.fmax.v8f32
  "llvm.intr.vp.reduce.fmax" (%f, %C, %mask, %evl) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  // CHECK: call float @llvm.vp.reduce.fmin.v8f32
  "llvm.intr.vp.reduce.fmin" (%f, %C, %mask, %evl) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32

  // CHECK: call <8 x i32> @llvm.vp.select.v8i32
  "llvm.intr.vp.select" (%mask, %A, %B, %evl) :
         (vector<8xi1>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vp.merge.v8i32
  "llvm.intr.vp.merge" (%mask, %A, %B, %evl) :
         (vector<8xi1>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>

  // CHECK: call void @llvm.vp.store.v8i32.p0
  "llvm.intr.vp.store" (%A, %iptr, %mask, %evl) :
         (vector<8xi32>, !llvm.ptr, vector<8xi1>, i32) -> ()
  // CHECK: call <8 x i32> @llvm.vp.load.v8i32.p0
  "llvm.intr.vp.load" (%iptr, %mask, %evl) :
         (!llvm.ptr, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call void @llvm.experimental.vp.strided.store.v8i32.p0.i32
  "llvm.intr.experimental.vp.strided.store" (%A, %iptr, %i, %mask, %evl) :
         (vector<8xi32>, !llvm.ptr, i32, vector<8xi1>, i32) -> ()
  // CHECK: call <8 x i32> @llvm.experimental.vp.strided.load.v8i32.p0.i32
  "llvm.intr.experimental.vp.strided.load" (%iptr, %i, %mask, %evl) :
         (!llvm.ptr, i32, vector<8xi1>, i32) -> vector<8xi32>

  // CHECK: call <8 x i32> @llvm.vp.trunc.v8i32.v8i64
  "llvm.intr.vp.trunc" (%E, %mask, %evl) :
         (vector<8xi64>, vector<8xi1>, i32) -> vector<8xi32>
  // CHECK: call <8 x i64> @llvm.vp.zext.v8i64.v8i32
  "llvm.intr.vp.zext" (%A, %mask, %evl) :
         (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>
  // CHECK: call <8 x i64> @llvm.vp.sext.v8i64.v8i32
  "llvm.intr.vp.sext" (%A, %mask, %evl) :
         (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>

  // CHECK: call <8 x float> @llvm.vp.fptrunc.v8f32.v8f64
  "llvm.intr.vp.fptrunc" (%F, %mask, %evl) :
         (vector<8xf64>, vector<8xi1>, i32) -> vector<8xf32>
  // CHECK: call <8 x double> @llvm.vp.fpext.v8f64.v8f32
  "llvm.intr.vp.fpext" (%C, %mask, %evl) :
         (vector<8xf32>, vector<8xi1>, i32) -> vector<8xf64>

  // CHECK: call <8 x i64> @llvm.vp.fptoui.v8i64.v8f64
  "llvm.intr.vp.fptoui" (%F, %mask, %evl) :
         (vector<8xf64>, vector<8xi1>, i32) -> vector<8xi64>
  // CHECK: call <8 x i64> @llvm.vp.fptosi.v8i64.v8f64
  "llvm.intr.vp.fptosi" (%F, %mask, %evl) :
         (vector<8xf64>, vector<8xi1>, i32) -> vector<8xi64>

  // CHECK: call <8 x i64> @llvm.vp.ptrtoint.v8i64.v8p0
  "llvm.intr.vp.ptrtoint" (%G, %mask, %evl) :
         (!llvm.vec<8 x !llvm.ptr>, vector<8xi1>, i32) -> vector<8xi64>
  // CHECK: call <8 x ptr> @llvm.vp.inttoptr.v8p0.v8i64
  "llvm.intr.vp.inttoptr" (%E, %mask, %evl) :
         (vector<8xi64>, vector<8xi1>, i32) -> !llvm.vec<8 x !llvm.ptr>
  llvm.return
}

// CHECK-LABEL: @vector_insert_extract
llvm.func @vector_insert_extract(%f256: vector<8xi32>, %f128: vector<4xi32>,
                                 %sv: vector<[4]xi32>) {
  // CHECK: call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v8i32
  %0 = llvm.intr.vector.insert %f256, %sv[0] :
              vector<8xi32> into vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32
  %1 = llvm.intr.vector.insert %f128, %sv[0] :
              vector<4xi32> into vector<[4]xi32>
  // CHECK: call <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32
  %2 = llvm.intr.vector.insert %f128, %1[4] :
              vector<4xi32> into vector<[4]xi32>
  // CHECK: call <8 x i32> @llvm.vector.insert.v8i32.v4i32
  %3 = llvm.intr.vector.insert %f128, %f256[4] :
              vector<4xi32> into vector<8xi32>
  // CHECK: call <8 x i32> @llvm.vector.extract.v8i32.nxv4i32
  %4 = llvm.intr.vector.extract %2[0] :
              vector<8xi32> from vector<[4]xi32>
  // CHECK: call <4 x i32> @llvm.vector.extract.v4i32.nxv4i32
  %5 = llvm.intr.vector.extract %2[0] :
              vector<4xi32> from vector<[4]xi32>
  // CHECK: call <2 x i32> @llvm.vector.extract.v2i32.v8i32
  %6 = llvm.intr.vector.extract %f256[6] :
              vector<2xi32> from vector<8xi32>
  llvm.return
}

// CHECK-LABEL: @vector_deinterleave2
llvm.func @vector_deinterleave2(%vec1: vector<4xf64>, %vec2: vector<[8]xi32>) {
  // CHECK: call { <2 x double>, <2 x double> } @llvm.vector.deinterleave2.v4f64(<4 x double> %{{.*}})
  %0 = "llvm.intr.vector.deinterleave2" (%vec1) :
              (vector<4xf64>) -> !llvm.struct<(vector<2xf64>, vector<2xf64>)>
  // CHECK: call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32> %{{.*}})
  %1 = "llvm.intr.vector.deinterleave2" (%vec2) :
              (vector<[8]xi32>) -> !llvm.struct<(vector<[4]xi32>, vector<[4]xi32>)>
  llvm.return
}

// CHECK-LABEL: @lifetime
llvm.func @lifetime(%p: !llvm.ptr) {
  // CHECK: call void @llvm.lifetime.start
  llvm.intr.lifetime.start 16, %p : !llvm.ptr
  // CHECK: call void @llvm.lifetime.end
  llvm.intr.lifetime.end 16, %p : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @invariant
llvm.func @invariant(%p: !llvm.ptr) {
  // CHECK: call ptr @llvm.invariant.start
  %1 = llvm.intr.invariant.start 16, %p : !llvm.ptr
  // CHECK: call void @llvm.invariant.end
  llvm.intr.invariant.end %1, 16, %p : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @invariant_group
llvm.func @invariant_group(%p: !llvm.ptr) {
  // CHECK: call ptr @llvm.launder.invariant.group
  %1 = llvm.intr.launder.invariant.group %p : !llvm.ptr
  // CHECK: call ptr @llvm.strip.invariant.group
  %2 = llvm.intr.strip.invariant.group %p : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @ssa_copy
llvm.func @ssa_copy(%arg: f32) -> f32 {
  // CHECK: call float @llvm.ssa.copy
  %0 = llvm.intr.ssa.copy %arg : f32
  llvm.return %0 : f32
}

// CHECK-LABEL: @experimental_constrained_fptrunc
llvm.func @experimental_constrained_fptrunc(%s: f64, %v: vector<4xf32>) {
  // CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(
  // CHECK: metadata !"round.towardzero"
  // CHECK: metadata !"fpexcept.ignore"
  %0 = llvm.intr.experimental.constrained.fptrunc %s towardzero ignore : f64 to f32
  // CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(
  // CHECK: metadata !"round.tonearest"
  // CHECK: metadata !"fpexcept.maytrap"
  %1 = llvm.intr.experimental.constrained.fptrunc %s tonearest maytrap : f64 to f32
  // CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(
  // CHECK: metadata !"round.upward"
  // CHECK: metadata !"fpexcept.strict"
  %2 = llvm.intr.experimental.constrained.fptrunc %s upward strict : f64 to f32
  // CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(
  // CHECK: metadata !"round.downward"
  // CHECK: metadata !"fpexcept.ignore"
  %3 = llvm.intr.experimental.constrained.fptrunc %s downward ignore : f64 to f32
  // CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(
  // CHECK: metadata !"round.tonearestaway"
  // CHECK: metadata !"fpexcept.ignore"
  %4 = llvm.intr.experimental.constrained.fptrunc %s tonearestaway ignore : f64 to f32
  // CHECK: call <4 x half> @llvm.experimental.constrained.fptrunc.v4f16.v4f32(
  // CHECK: metadata !"round.upward"
  // CHECK: metadata !"fpexcept.strict"
  %5 = llvm.intr.experimental.constrained.fptrunc %v upward strict : vector<4xf32> to vector<4xf16>
  llvm.return
}

// Check that intrinsics are declared with appropriate types.
// CHECK-DAG: declare float @llvm.fma.f32(float, float, float)
// CHECK-DAG: declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
// CHECK-DAG: declare float @llvm.fmuladd.f32(float, float, float)
// CHECK-DAG: declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
// CHECK-DAG: declare void @llvm.prefetch.p0(ptr nocapture readonly, i32 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare i1 @llvm.is.fpclass.f32(float, i32 immarg)
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
// CHECK-DAG: declare float @llvm.rint.f32(float)
// CHECK-DAG: declare double @llvm.rint.f64(double)
// CHECK-DAG: declare <8 x float> @llvm.rint.v8f32(<8 x float>)
// CHECK-DAG: declare <8 x double> @llvm.rint.v8f64(<8 x double>)
// CHECK-DAG: declare float @llvm.nearbyint.f32(float)
// CHECK-DAG: declare double @llvm.nearbyint.f64(double)
// CHECK-DAG: declare <8 x float> @llvm.nearbyint.v8f32(<8 x float>)
// CHECK-DAG: declare <8 x double> @llvm.nearbyint.v8f64(<8 x double>)
// CHECK-DAG: declare i32 @llvm.lround.i32.f32(float)
// CHECK-DAG: declare i64 @llvm.lround.i64.f32(float)
// CHECK-DAG: declare i32 @llvm.lround.i32.f64(double)
// CHECK-DAG: declare i64 @llvm.lround.i64.f64(double)
// CHECK-DAG: declare i64 @llvm.llround.i64.f32(float)
// CHECK-DAG: declare i64 @llvm.llround.i64.f64(double)
// CHECK-DAG: declare i32 @llvm.lrint.i32.f32(float)
// CHECK-DAG: declare i64 @llvm.lrint.i64.f32(float)
// CHECK-DAG: declare i32 @llvm.lrint.i32.f64(double)
// CHECK-DAG: declare i64 @llvm.lrint.i64.f64(double)
// CHECK-DAG: declare i64 @llvm.llrint.i64.f32(float)
// CHECK-DAG: declare i64 @llvm.llrint.i64.f64(double)
// CHECK-DAG: declare <12 x float> @llvm.matrix.multiply.v12f32.v64f32.v48f32(<64 x float>, <48 x float>, i32 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare <48 x float> @llvm.matrix.transpose.v48f32(<48 x float>, i32 immarg, i32 immarg)
// CHECK-DAG: declare <48 x float> @llvm.matrix.column.major.load.v48f32.i64(ptr nocapture, i64, i1 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare void @llvm.matrix.column.major.store.v48f32.i64(<48 x float>, ptr nocapture writeonly, i64, i1 immarg, i32 immarg, i32 immarg)
// CHECK-DAG: declare <7 x i1> @llvm.get.active.lane.mask.v7i1.i64(i64, i64)
// CHECK-DAG: declare <7 x float> @llvm.masked.load.v7f32.p0(ptr nocapture, i32 immarg, <7 x i1>, <7 x float>)
// CHECK-DAG: declare void @llvm.masked.store.v7f32.p0(<7 x float>, ptr nocapture, i32 immarg, <7 x i1>)
// CHECK-DAG: declare <7 x float> @llvm.masked.gather.v7f32.v7p0(<7 x ptr>, i32 immarg, <7 x i1>, <7 x float>)
// CHECK-DAG: declare void @llvm.masked.scatter.v7f32.v7p0(<7 x float>, <7 x ptr>, i32 immarg, <7 x i1>)
// CHECK-DAG: declare <7 x float> @llvm.masked.expandload.v7f32(ptr nocapture, <7 x i1>, <7 x float>)
// CHECK-DAG: declare void @llvm.masked.compressstore.v7f32(<7 x float>, ptr nocapture, <7 x i1>)
// CHECK-DAG: declare void @llvm.var.annotation.p0.p0(ptr, ptr, ptr, i32, ptr)
// CHECK-DAG: declare ptr @llvm.ptr.annotation.p0.p0(ptr, ptr, ptr, i32, ptr)
// CHECK-DAG: declare i16 @llvm.annotation.i16.p0(i16, ptr, ptr, i32)
// CHECK-DAG: declare void @llvm.trap()
// CHECK-DAG: declare void @llvm.debugtrap()
// CHECK-DAG: declare void @llvm.ubsantrap(i8 immarg)
// CHECK-DAG: declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
// CHECK-DAG: declare void @llvm.memcpy.inline.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
// CHECK-DAG: declare void @llvm.memcpy.inline.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
// CHECK-DAG: declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.sadd.with.overflow.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.uadd.with.overflow.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.ssub.with.overflow.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.usub.with.overflow.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
// CHECK-DAG: declare { <8 x i32>, <8 x i1> } @llvm.umul.with.overflow.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare i32 @llvm.sadd.sat.i32(i32, i32)
// CHECK-DAG: declare <8 x i32> @llvm.sadd.sat.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare i32 @llvm.uadd.sat.i32(i32, i32)
// CHECK-DAG: declare <8 x i32> @llvm.uadd.sat.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare i32 @llvm.ssub.sat.i32(i32, i32)
// CHECK-DAG: declare <8 x i32> @llvm.ssub.sat.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare i32 @llvm.usub.sat.i32(i32, i32)
// CHECK-DAG: declare <8 x i32> @llvm.usub.sat.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare i32 @llvm.sshl.sat.i32(i32, i32)
// CHECK-DAG: declare <8 x i32> @llvm.sshl.sat.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare i32 @llvm.ushl.sat.i32(i32, i32)
// CHECK-DAG: declare <8 x i32> @llvm.ushl.sat.v8i32(<8 x i32>, <8 x i32>)
// CHECK-DAG: declare i1 @llvm.is.constant.i32(i32)
// CHECK-DAG: declare i32 @llvm.expect.i32(i32, i32)
// CHECK-DAG: declare i16 @llvm.expect.with.probability.i16(i16, i16, double immarg)
// CHECK-DAG: declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)
// CHECK-DAG: declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
// CHECK-DAG: declare ptr @llvm.coro.begin(token, ptr writeonly)
// CHECK-DAG: declare i64 @llvm.coro.size.i64()
// CHECK-DAG: declare i32 @llvm.coro.size.i32()
// CHECK-DAG: declare token @llvm.coro.save(ptr)
// CHECK-DAG: declare i8 @llvm.coro.suspend(token, i1)
// CHECK-DAG: declare i1 @llvm.coro.end(ptr, i1, token)
// CHECK-DAG: declare ptr @llvm.coro.free(token, ptr nocapture readonly)
// CHECK-DAG: declare void @llvm.coro.resume(ptr)
// CHECK-DAG: declare ptr @llvm.coro.promise(ptr nocapture, i32, i1)
// CHECK-DAG: declare <8 x i32> @llvm.vp.add.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.sub.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.mul.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.sdiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.udiv.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.srem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.urem.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.ashr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.lshr.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.shl.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.or.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.and.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.xor.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.smax.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.smin.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.umax.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.umin.v8i32(<8 x i32>, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.fadd.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.fsub.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.fmul.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.fdiv.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.frem.v8f32(<8 x float>, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.fneg.v8f32(<8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.fma.v8f32(<8 x float>, <8 x float>, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.add.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.mul.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.and.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.or.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.xor.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.smax.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.smin.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.umax.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare i32 @llvm.vp.reduce.umin.v8i32(i32, <8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare float @llvm.vp.reduce.fadd.v8f32(float, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare float @llvm.vp.reduce.fmul.v8f32(float, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare float @llvm.vp.reduce.fmax.v8f32(float, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare float @llvm.vp.reduce.fmin.v8f32(float, <8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.select.v8i32(<8 x i1>, <8 x i32>, <8 x i32>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.merge.v8i32(<8 x i1>, <8 x i32>, <8 x i32>, i32)
// CHECK-DAG: declare void @llvm.experimental.vp.strided.store.v8i32.p0.i32(<8 x i32>, ptr nocapture, i32, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.experimental.vp.strided.load.v8i32.p0.i32(ptr nocapture, i32, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i32> @llvm.vp.trunc.v8i32.v8i64(<8 x i64>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i64> @llvm.vp.zext.v8i64.v8i32(<8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i64> @llvm.vp.sext.v8i64.v8i32(<8 x i32>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x float> @llvm.vp.fptrunc.v8f32.v8f64(<8 x double>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x double> @llvm.vp.fpext.v8f64.v8f32(<8 x float>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i64> @llvm.vp.fptoui.v8i64.v8f64(<8 x double>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i64> @llvm.vp.fptosi.v8i64.v8f64(<8 x double>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x i64> @llvm.vp.ptrtoint.v8i64.v8p0(<8 x ptr>, <8 x i1>, i32)
// CHECK-DAG: declare <8 x ptr> @llvm.vp.inttoptr.v8p0.v8i64(<8 x i64>, <8 x i1>, i32)
// CHECK-DAG: declare <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v8i32(<vscale x 4 x i32>, <8 x i32>, i64 immarg)
// CHECK-DAG: declare <vscale x 4 x i32> @llvm.vector.insert.nxv4i32.v4i32(<vscale x 4 x i32>, <4 x i32>, i64 immarg)
// CHECK-DAG: declare <8 x i32> @llvm.vector.insert.v8i32.v4i32(<8 x i32>, <4 x i32>, i64 immarg)
// CHECK-DAG: declare <8 x i32> @llvm.vector.extract.v8i32.nxv4i32(<vscale x 4 x i32>, i64 immarg)
// CHECK-DAG: declare <4 x i32> @llvm.vector.extract.v4i32.nxv4i32(<vscale x 4 x i32>, i64 immarg)
// CHECK-DAG: declare <2 x i32> @llvm.vector.extract.v2i32.v8i32(<8 x i32>, i64 immarg)
// CHECK-DAG: declare { <2 x double>, <2 x double> } @llvm.vector.deinterleave2.v4f64(<4 x double>)
// CHECK-DAG: declare { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32>)
// CHECK-DAG: declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
// CHECK-DAG: declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
// CHECK-DAG: declare ptr @llvm.invariant.start.p0(i64 immarg, ptr nocapture)
// CHECK-DAG: declare void @llvm.invariant.end.p0(ptr, i64 immarg, ptr nocapture)

// CHECK-DAG: declare float @llvm.ssa.copy.f32(float returned)
// CHECK-DAG: declare ptr @llvm.stacksave.p0()
// CHECK-DAG: declare ptr addrspace(1) @llvm.stacksave.p1()
// CHECK-DAG: declare void @llvm.stackrestore.p0(ptr)
// CHECK-DAG: declare void @llvm.stackrestore.p1(ptr addrspace(1))
// CHECK-DAG: declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
// CHECK-DAG: declare <4 x half> @llvm.experimental.constrained.fptrunc.v4f16.v4f32(<4 x float>, metadata, metadata)
