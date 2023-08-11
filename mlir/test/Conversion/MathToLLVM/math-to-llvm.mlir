// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(func.func(convert-math-to-llvm))" | FileCheck %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm="filter-dialects=math" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @ops
func.func @ops(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: f64) {
  // CHECK: = llvm.intr.exp(%{{.*}}) : (f32) -> f32
  %0 = math.exp %arg0 : f32
  // CHECK: = llvm.intr.exp2(%{{.*}}) : (f32) -> f32
  %1 = math.exp2 %arg0 : f32
  // CHECK: = llvm.intr.sqrt(%{{.*}}) : (f32) -> f32
  %2 = math.sqrt %arg0 : f32
  // CHECK: = llvm.intr.sqrt(%{{.*}}) : (f64) -> f64
  %3 = math.sqrt %arg4 : f64
  func.return
}

// -----

func.func @absi(%arg0: i32) -> i32 {
  // CHECK: = "llvm.intr.abs"(%{{.*}}) <{is_int_min_poison = false}> : (i32) -> i32
  %0 = math.absi %arg0 : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func @log1p(
// CHECK-SAME: f32
func.func @log1p(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[ONE]], %arg0 : f32
  // CHECK: %[[LOG:.*]] = llvm.intr.log(%[[ADD]]) : (f32) -> f32
  %0 = math.log1p %arg0 : f32
  func.return
}

// -----

// CHECK-LABEL: func @log1p_fmf(
// CHECK-SAME: f32
func.func @log1p_fmf(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[ONE]], %arg0 {fastmathFlags = #llvm.fastmath<fast>} : f32
  // CHECK: %[[LOG:.*]] = llvm.intr.log(%[[ADD]]) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %0 = math.log1p %arg0 fastmath<fast> : f32
  func.return
}

// -----

// CHECK-LABEL: func @log1p_2dvector(
func.func @log1p_2dvector(%arg0 : vector<4x3xf32>) {
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<3xf32>) : vector<3xf32>
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[ONE]], %[[EXTRACT]] : vector<3xf32>
  // CHECK: %[[LOG:.*]] = llvm.intr.log(%[[ADD]]) : (vector<3xf32>) -> vector<3xf32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[LOG]], %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  %0 = math.log1p %arg0 : vector<4x3xf32>
  func.return
}

// -----

// CHECK-LABEL: func @log1p_2dvector_fmf(
func.func @log1p_2dvector_fmf(%arg0 : vector<4x3xf32>) {
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<3xf32>) : vector<3xf32>
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[ONE]], %[[EXTRACT]] {fastmathFlags = #llvm.fastmath<fast>} : vector<3xf32>
  // CHECK: %[[LOG:.*]] = llvm.intr.log(%[[ADD]]) {fastmathFlags = #llvm.fastmath<fast>} : (vector<3xf32>) -> vector<3xf32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[LOG]], %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  %0 = math.log1p %arg0 fastmath<fast> : vector<4x3xf32>
  func.return
}

// -----

// CHECK-LABEL: func @expm1(
// CHECK-SAME: f32
func.func @expm1(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[EXP:.*]] = llvm.intr.exp(%arg0) : (f32) -> f32
  // CHECK: %[[SUB:.*]] = llvm.fsub %[[EXP]], %[[ONE]] : f32
  %0 = math.expm1 %arg0 : f32
  func.return
}

// -----

// CHECK-LABEL: func @expm1_fmf(
// CHECK-SAME: f32
func.func @expm1_fmf(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[EXP:.*]] = llvm.intr.exp(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  // CHECK: %[[SUB:.*]] = llvm.fsub %[[EXP]], %[[ONE]] {fastmathFlags = #llvm.fastmath<fast>} : f32
  %0 = math.expm1 %arg0 fastmath<fast> : f32
  func.return
}

// -----

// CHECK-LABEL: func @expm1_vector(
// CHECK-SAME: vector<4xf32>
func.func @expm1_vector(%arg0 : vector<4xf32>) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[EXP:.*]] = llvm.intr.exp(%arg0) : (vector<4xf32>) -> vector<4xf32>
  // CHECK: %[[SUB:.*]] = llvm.fsub %[[EXP]], %[[ONE]] : vector<4xf32>
  %0 = math.expm1 %arg0 : vector<4xf32>
  func.return
}

// -----

// CHECK-LABEL: func @expm1_vector_fmf(
// CHECK-SAME: vector<4xf32>
func.func @expm1_vector_fmf(%arg0 : vector<4xf32>) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[EXP:.*]] = llvm.intr.exp(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<4xf32>) -> vector<4xf32>
  // CHECK: %[[SUB:.*]] = llvm.fsub %[[EXP]], %[[ONE]] {fastmathFlags = #llvm.fastmath<fast>} : vector<4xf32>
  %0 = math.expm1 %arg0 fastmath<fast> : vector<4xf32>
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt(
// CHECK-SAME: f32
func.func @rsqrt(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%arg0) : (f32) -> f32
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : f32
  %0 = math.rsqrt %arg0 : f32
  func.return
}

// -----

// CHECK-LABEL: func @sine(
// CHECK-SAME: f32
func.func @sine(%arg0 : f32) {
  // CHECK: llvm.intr.sin(%arg0) : (f32) -> f32
  %0 = math.sin %arg0 : f32
  func.return
}

// -----

// CHECK-LABEL: func @ctlz(
// CHECK-SAME: i32
func.func @ctlz(%arg0 : i32) {
  // CHECK: "llvm.intr.ctlz"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
  %0 = math.ctlz %arg0 : i32
  func.return
}

// -----

// CHECK-LABEL: func @cttz(
// CHECK-SAME: i32
func.func @cttz(%arg0 : i32) {
  // CHECK: "llvm.intr.cttz"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
  %0 = math.cttz %arg0 : i32
  func.return
}

// -----

// CHECK-LABEL: func @cttz_vec(
// CHECK-SAME: i32
func.func @cttz_vec(%arg0 : vector<4xi32>) {
  // CHECK: "llvm.intr.cttz"(%arg0) <{is_zero_poison = false}> : (vector<4xi32>) -> vector<4xi32>
  %0 = math.cttz %arg0 : vector<4xi32>
  func.return
}

// -----

// CHECK-LABEL: func @ctpop(
// CHECK-SAME: i32
func.func @ctpop(%arg0 : i32) {
  // CHECK: llvm.intr.ctpop(%arg0) : (i32) -> i32
  %0 = math.ctpop %arg0 : i32
  func.return
}

// -----

// CHECK-LABEL: func @ctpop_vector(
// CHECK-SAME: vector<3xi32>
func.func @ctpop_vector(%arg0 : vector<3xi32>) {
  // CHECK: llvm.intr.ctpop(%arg0) : (vector<3xi32>) -> vector<3xi32>
  %0 = math.ctpop %arg0 : vector<3xi32>
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt_double(
// CHECK-SAME: f64
func.func @rsqrt_double(%arg0 : f64) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
  // CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%arg0) : (f64) -> f64
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : f64
  %0 = math.rsqrt %arg0 : f64
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt_double_fmf(
// CHECK-SAME: f64
func.func @rsqrt_double_fmf(%arg0 : f64) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
  // CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f64) -> f64
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] {fastmathFlags = #llvm.fastmath<fast>} : f64
  %0 = math.rsqrt %arg0 fastmath<fast> : f64
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt_vector(
// CHECK-SAME: vector<4xf32>
func.func @rsqrt_vector(%arg0 : vector<4xf32>) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%arg0) : (vector<4xf32>) -> vector<4xf32>
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : vector<4xf32>
  %0 = math.rsqrt %arg0 : vector<4xf32>
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt_vector_fmf(
// CHECK-SAME: vector<4xf32>
func.func @rsqrt_vector_fmf(%arg0 : vector<4xf32>) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<4xf32>) -> vector<4xf32>
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] {fastmathFlags = #llvm.fastmath<fast>} : vector<4xf32>
  %0 = math.rsqrt %arg0 fastmath<fast> : vector<4xf32>
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt_multidim_vector(
func.func @rsqrt_multidim_vector(%arg0 : vector<4x3xf32>) {
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<3xf32>) : vector<3xf32>
  // CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%[[EXTRACT]]) : (vector<3xf32>) -> vector<3xf32>
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : vector<3xf32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[DIV]], %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  %0 = math.rsqrt %arg0 : vector<4x3xf32>
  func.return
}

// -----

// CHECK-LABEL: func @fpowi(
// CHECK-SAME: f64
func.func @fpowi(%arg0 : f64, %arg1 : i32) {
  // CHECK: llvm.intr.powi(%arg0, %arg1) : (f64, i32) -> f64
  %0 = math.fpowi %arg0, %arg1 : f64, i32
  func.return
}


// -----

// CHECK-LABEL: func @powf(
// CHECK-SAME: f64
func.func @powf(%arg0 : f64) {
  // CHECK: %[[POWF:.*]] = llvm.intr.pow(%arg0, %arg0) : (f64, f64) -> f64
  %0 = math.powf %arg0, %arg0 : f64
  func.return
}

// -----

// CHECK-LABEL: func @round(
// CHECK-SAME: f32
func.func @round(%arg0 : f32) {
  // CHECK: llvm.intr.round(%arg0) : (f32) -> f32
  %0 = math.round %arg0 : f32
  func.return
}

// -----

// CHECK-LABEL: func @roundeven(
// CHECK-SAME: f32
func.func @roundeven(%arg0 : f32) {
  // CHECK: llvm.intr.roundeven(%arg0) : (f32) -> f32
  %0 = math.roundeven %arg0 : f32
  func.return
}

// -----

// CHECK-LABEL: func @trunc(
// CHECK-SAME: f32
func.func @trunc(%arg0 : f32) {
  // CHECK: llvm.intr.trunc(%arg0) : (f32) -> f32
  %0 = math.trunc %arg0 : f32
  func.return
}

// -----

// CHECK-LABEL: func @fastmath(
// CHECK-SAME: f32
func.func @fastmath(%arg0 : f32, %arg1 : vector<4xf32>) {
  // CHECK: llvm.intr.trunc(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %0 = math.trunc %arg0 fastmath<fast> : f32
  // CHECK: llvm.intr.pow(%arg0, %arg0) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32
  %1 = math.powf %arg0, %arg0 fastmath<afn> : f32
  // CHECK: llvm.intr.sqrt(%arg0) : (f32) -> f32
  %2 = math.sqrt %arg0 fastmath<none> : f32
  // CHECK: llvm.intr.fma(%arg0, %arg0, %arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32, f32, f32) -> f32
  %3 = math.fma %arg0, %arg0, %arg0 fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn> : f32
  func.return
}
