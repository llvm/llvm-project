// RUN: mlir-opt %s -split-input-file -convert-math-to-llvm | FileCheck %s

// CHECK-LABEL: @ops
func @ops(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: f64) {
// CHECK: = "llvm.intr.exp"(%{{.*}}) : (f32) -> f32
  %13 = math.exp %arg0 : f32
// CHECK: = "llvm.intr.exp2"(%{{.*}}) : (f32) -> f32
  %14 = math.exp2 %arg0 : f32
// CHECK: = "llvm.intr.sqrt"(%{{.*}}) : (f32) -> f32
  %19 = math.sqrt %arg0 : f32
// CHECK: = "llvm.intr.sqrt"(%{{.*}}) : (f64) -> f64
  %20 = math.sqrt %arg4 : f64
  std.return
}

// -----

// CHECK-LABEL: func @log1p(
// CHECK-SAME: f32
func @log1p(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[ONE]], %arg0 : f32
  // CHECK: %[[LOG:.*]] = "llvm.intr.log"(%[[ADD]]) : (f32) -> f32
  %0 = math.log1p %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @log1p_2dvector(
func @log1p_2dvector(%arg0 : vector<4x3xf32>) {
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<3xf32>) : vector<3xf32>
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[ONE]], %[[EXTRACT]] : vector<3xf32>
  // CHECK: %[[LOG:.*]] = "llvm.intr.log"(%[[ADD]]) : (vector<3xf32>) -> vector<3xf32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[LOG]], %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  %0 = math.log1p %arg0 : vector<4x3xf32>
  std.return
}

// -----

// CHECK-LABEL: func @expm1(
// CHECK-SAME: f32
func @expm1(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[EXP:.*]] = "llvm.intr.exp"(%arg0) : (f32) -> f32
  // CHECK: %[[SUB:.*]] = llvm.fsub %[[EXP]], %[[ONE]] : f32
  %0 = math.expm1 %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt(
// CHECK-SAME: f32
func @rsqrt(%arg0 : f32) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%arg0) : (f32) -> f32
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : f32
  %0 = math.rsqrt %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @sine(
// CHECK-SAME: f32
func @sine(%arg0 : f32) {
  // CHECK: "llvm.intr.sin"(%arg0) : (f32) -> f32
  %0 = math.sin %arg0 : f32
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt_double(
// CHECK-SAME: f64
func @rsqrt_double(%arg0 : f64) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%arg0) : (f64) -> f64
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : f64
  %0 = math.rsqrt %arg0 : f64
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt_vector(
// CHECK-SAME: vector<4xf32>
func @rsqrt_vector(%arg0 : vector<4xf32>) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%arg0) : (vector<4xf32>) -> vector<4xf32>
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : vector<4xf32>
  %0 = math.rsqrt %arg0 : vector<4xf32>
  std.return
}

// -----

// CHECK-LABEL: func @rsqrt_multidim_vector(
func @rsqrt_multidim_vector(%arg0 : vector<4x3xf32>) {
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<3xf32>) : vector<3xf32>
  // CHECK: %[[SQRT:.*]] = "llvm.intr.sqrt"(%[[EXTRACT]]) : (vector<3xf32>) -> vector<3xf32>
  // CHECK: %[[DIV:.*]] = llvm.fdiv %[[ONE]], %[[SQRT]] : vector<3xf32>
  // CHECK: %[[INSERT:.*]] = llvm.insertvalue %[[DIV]], %{{.*}}[0] : !llvm.array<4 x vector<3xf32>>
  %0 = math.rsqrt %arg0 : vector<4x3xf32>
  std.return
}

// -----

// CHECK-LABEL: func @powf(
// CHECK-SAME: f64
func @powf(%arg0 : f64) {
  // CHECK: %[[POWF:.*]] = "llvm.intr.pow"(%arg0, %arg0) : (f64, f64) -> f64
  %0 = math.powf %arg0, %arg0 : f64
  std.return
}

