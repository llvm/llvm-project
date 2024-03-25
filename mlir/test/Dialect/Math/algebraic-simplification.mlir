// RUN: mlir-opt %s -test-math-algebraic-simplification | FileCheck %s --dump-input=always

// CHECK-LABEL: @pow_noop
func.func @pow_noop(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: return %arg0, %arg1
  %c = arith.constant 1.0 : f32
  %v = arith.constant dense <1.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_square
func.func @pow_square(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[SCALAR:.*]] = arith.mulf %arg0, %arg0
  // CHECK: %[[VECTOR:.*]] = arith.mulf %arg1, %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant 2.0 : f32
  %v = arith.constant dense <2.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_cube
func.func @pow_cube(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[TMP_S:.*]] = arith.mulf %arg0, %arg0
  // CHECK: %[[SCALAR:.*]] = arith.mulf %arg0, %[[TMP_S]]
  // CHECK: %[[TMP_V:.*]] = arith.mulf %arg1, %arg1
  // CHECK: %[[VECTOR:.*]] = arith.mulf %arg1, %[[TMP_V]]
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant 3.0 : f32
  %v = arith.constant dense <3.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_recip
func.func @pow_recip(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK-DAG: %[[CST_S:.*]] = arith.constant 1.0{{.*}} : f32
  // CHECK-DAG: %[[CST_V:.*]] = arith.constant dense<1.0{{.*}}> : vector<4xf32>
  // CHECK: %[[SCALAR:.*]] = arith.divf %[[CST_S]], %arg0
  // CHECK: %[[VECTOR:.*]] = arith.divf %[[CST_V]], %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant -1.0 : f32
  %v = arith.constant dense <-1.0> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_sqrt
func.func @pow_sqrt(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[SCALAR:.*]] = math.sqrt %arg0
  // CHECK: %[[VECTOR:.*]] = math.sqrt %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant 0.5 : f32
  %v = arith.constant dense <0.5> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_rsqrt
func.func @pow_rsqrt(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[SCALAR:.*]] = math.rsqrt %arg0
  // CHECK: %[[VECTOR:.*]] = math.rsqrt %arg1
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant -0.5 : f32
  %v = arith.constant dense <-0.5> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @pow_0_75
func.func @pow_0_75(%arg0: f32, %arg1 : vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[SQRT1S:.*]] = math.sqrt %arg0
  // CHECK: %[[SQRT2S:.*]] = math.sqrt %[[SQRT1S]]
  // CHECK: %[[SCALAR:.*]] = arith.mulf %[[SQRT1S]], %[[SQRT2S]]
  // CHECK: %[[SQRT1V:.*]] = math.sqrt %arg1
  // CHECK: %[[SQRT2V:.*]] = math.sqrt %[[SQRT1V]]
  // CHECK: %[[VECTOR:.*]] = arith.mulf %[[SQRT1V]], %[[SQRT2V]]
  // CHECK: return %[[SCALAR]], %[[VECTOR]]
  %c = arith.constant 0.75 : f32
  %v = arith.constant dense <0.75> : vector<4xf32>
  %0 = math.powf %arg0, %c : f32
  %1 = math.powf %arg1, %v : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @ipowi_zero_exp(
// CHECK-SAME: %[[ARG0:.+]]: i32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xi32>
// CHECK-SAME: -> (i32, vector<4xi32>) {
func.func @ipowi_zero_exp(%arg0: i32, %arg1: vector<4xi32>) -> (i32, vector<4xi32>) {
  // CHECK: %[[CST_S:.*]] = arith.constant 1 : i32
  // CHECK: %[[CST_V:.*]] = arith.constant dense<1> : vector<4xi32>
  // CHECK: return %[[CST_S]], %[[CST_V]]
  %c = arith.constant 0 : i32
  %v = arith.constant dense <0> : vector<4xi32>
  %0 = math.ipowi %arg0, %c : i32
  %1 = math.ipowi %arg1, %v : vector<4xi32>
  return %0, %1 : i32, vector<4xi32>
}

// CHECK-LABEL: @ipowi_exp_one(
// CHECK-SAME: %[[ARG0:.+]]: i32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xi32>
// CHECK-SAME: -> (i32, vector<4xi32>, i32, vector<4xi32>) {
func.func @ipowi_exp_one(%arg0: i32, %arg1: vector<4xi32>) -> (i32, vector<4xi32>, i32, vector<4xi32>) {
  // CHECK-DAG: %[[CST_S:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[CST_V:.*]] = arith.constant dense<1> : vector<4xi32>
  // CHECK: %[[SCALAR:.*]] = arith.divsi %[[CST_S]], %[[ARG0]]
  // CHECK: %[[VECTOR:.*]] = arith.divsi %[[CST_V]], %[[ARG1]]
  // CHECK: return %[[ARG0]], %[[ARG1]], %[[SCALAR]], %[[VECTOR]]
  %c1 = arith.constant 1 : i32
  %v1 = arith.constant dense <1> : vector<4xi32>
  %0 = math.ipowi %arg0, %c1 : i32
  %1 = math.ipowi %arg1, %v1 : vector<4xi32>
  %cm1 = arith.constant -1 : i32
  %vm1 = arith.constant dense <-1> : vector<4xi32>
  %2 = math.ipowi %arg0, %cm1 : i32
  %3 = math.ipowi %arg1, %vm1 : vector<4xi32>
  return %0, %1, %2, %3 : i32, vector<4xi32>, i32, vector<4xi32>
}

// CHECK-LABEL: @ipowi_exp_two(
// CHECK-SAME: %[[ARG0:.+]]: i32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xi32>
// CHECK-SAME: -> (i32, vector<4xi32>, i32, vector<4xi32>) {
func.func @ipowi_exp_two(%arg0: i32, %arg1: vector<4xi32>) -> (i32, vector<4xi32>, i32, vector<4xi32>) {
  // CHECK-DAG: %[[CST_S:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[CST_V:.*]] = arith.constant dense<1> : vector<4xi32>
  // CHECK: %[[SCALAR0:.*]] = arith.muli %[[ARG0]], %[[ARG0]]
  // CHECK: %[[VECTOR0:.*]] = arith.muli %[[ARG1]], %[[ARG1]]
  // CHECK: %[[SCALAR1:.*]] = arith.divsi %[[CST_S]], %[[ARG0]]
  // CHECK: %[[SMUL:.*]] = arith.muli %[[SCALAR1]], %[[SCALAR1]]
  // CHECK: %[[VECTOR1:.*]] = arith.divsi %[[CST_V]], %[[ARG1]]
  // CHECK: %[[VMUL:.*]] = arith.muli %[[VECTOR1]], %[[VECTOR1]]
  // CHECK: return %[[SCALAR0]], %[[VECTOR0]], %[[SMUL]], %[[VMUL]]
  %c1 = arith.constant 2 : i32
  %v1 = arith.constant dense <2> : vector<4xi32>
  %0 = math.ipowi %arg0, %c1 : i32
  %1 = math.ipowi %arg1, %v1 : vector<4xi32>
  %cm1 = arith.constant -2 : i32
  %vm1 = arith.constant dense <-2> : vector<4xi32>
  %2 = math.ipowi %arg0, %cm1 : i32
  %3 = math.ipowi %arg1, %vm1 : vector<4xi32>
  return %0, %1, %2, %3 : i32, vector<4xi32>, i32, vector<4xi32>
}

// CHECK-LABEL: @ipowi_exp_three(
// CHECK-SAME: %[[ARG0:.+]]: i32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xi32>
// CHECK-SAME: -> (i32, vector<4xi32>, i32, vector<4xi32>) {
func.func @ipowi_exp_three(%arg0: i32, %arg1: vector<4xi32>) -> (i32, vector<4xi32>, i32, vector<4xi32>) {
  // CHECK-DAG: %[[CST_S:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[CST_V:.*]] = arith.constant dense<1> : vector<4xi32>
  // CHECK: %[[SMUL0:.*]] = arith.muli %[[ARG0]], %[[ARG0]]
  // CHECK: %[[SCALAR0:.*]] = arith.muli %[[SMUL0]], %[[ARG0]]
  // CHECK: %[[VMUL0:.*]] = arith.muli %[[ARG1]], %[[ARG1]]
  // CHECK: %[[VECTOR0:.*]] = arith.muli %[[VMUL0]], %[[ARG1]]
  // CHECK: %[[SCALAR1:.*]] = arith.divsi %[[CST_S]], %[[ARG0]]
  // CHECK: %[[SMUL1:.*]] = arith.muli %[[SCALAR1]], %[[SCALAR1]]
  // CHECK: %[[SMUL2:.*]] = arith.muli %[[SMUL1]], %[[SCALAR1]]
  // CHECK: %[[VECTOR1:.*]] = arith.divsi %[[CST_V]], %[[ARG1]]
  // CHECK: %[[VMUL1:.*]] = arith.muli %[[VECTOR1]], %[[VECTOR1]]
  // CHECK: %[[VMUL2:.*]] = arith.muli %[[VMUL1]], %[[VECTOR1]]
  // CHECK: return %[[SCALAR0]], %[[VECTOR0]], %[[SMUL2]], %[[VMUL2]]
  %c1 = arith.constant 3 : i32
  %v1 = arith.constant dense <3> : vector<4xi32>
  %0 = math.ipowi %arg0, %c1 : i32
  %1 = math.ipowi %arg1, %v1 : vector<4xi32>
  %cm1 = arith.constant -3 : i32
  %vm1 = arith.constant dense <-3> : vector<4xi32>
  %2 = math.ipowi %arg0, %cm1 : i32
  %3 = math.ipowi %arg1, %vm1 : vector<4xi32>
  return %0, %1, %2, %3 : i32, vector<4xi32>, i32, vector<4xi32>
}

// CHECK-LABEL: @fpowi_zero_exp(
// CHECK-SAME: %[[ARG0:.+]]: f32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xf32>
// CHECK-SAME: -> (f32, vector<4xf32>) {
func.func @fpowi_zero_exp(%arg0: f32, %arg1: vector<4xf32>) -> (f32, vector<4xf32>) {
  // CHECK: %[[CST_S:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[CST_V:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
  // CHECK: return %[[CST_S]], %[[CST_V]]
  %c = arith.constant 0 : i32
  %v = arith.constant dense <0> : vector<4xi32>
  %0 = math.fpowi %arg0, %c : f32, i32
  %1 = math.fpowi %arg1, %v : vector<4xf32>, vector<4xi32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: @fpowi_exp_one(
// CHECK-SAME: %[[ARG0:.+]]: f32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xf32>
// CHECK-SAME: -> (f32, vector<4xf32>, f32, vector<4xf32>) {
func.func @fpowi_exp_one(%arg0: f32, %arg1: vector<4xf32>) -> (f32, vector<4xf32>, f32, vector<4xf32>) {
  // CHECK-DAG: %[[CST_S:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[CST_V:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
  // CHECK: %[[SCALAR:.*]] = arith.divf %[[CST_S]], %[[ARG0]]
  // CHECK: %[[VECTOR:.*]] = arith.divf %[[CST_V]], %[[ARG1]]
  // CHECK: return %[[ARG0]], %[[ARG1]], %[[SCALAR]], %[[VECTOR]]
  %c1 = arith.constant 1 : i32
  %v1 = arith.constant dense <1> : vector<4xi32>
  %0 = math.fpowi %arg0, %c1 : f32, i32
  %1 = math.fpowi %arg1, %v1 : vector<4xf32>, vector<4xi32>
  %cm1 = arith.constant -1 : i32
  %vm1 = arith.constant dense <-1> : vector<4xi32>
  %2 = math.fpowi %arg0, %cm1 : f32, i32
  %3 = math.fpowi %arg1, %vm1 : vector<4xf32>, vector<4xi32>
  return %0, %1, %2, %3 : f32, vector<4xf32>, f32, vector<4xf32>
}

// CHECK-LABEL: @fpowi_exp_two(
// CHECK-SAME: %[[ARG0:.+]]: f32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xf32>
// CHECK-SAME: -> (f32, vector<4xf32>, f32, vector<4xf32>) {
func.func @fpowi_exp_two(%arg0: f32, %arg1: vector<4xf32>) -> (f32, vector<4xf32>, f32, vector<4xf32>) {
  // CHECK-DAG: %[[CST_S:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[CST_V:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
  // CHECK: %[[SCALAR0:.*]] = arith.mulf %[[ARG0]], %[[ARG0]]
  // CHECK: %[[VECTOR0:.*]] = arith.mulf %[[ARG1]], %[[ARG1]]
  // CHECK: %[[SCALAR1:.*]] = arith.divf %[[CST_S]], %[[ARG0]]
  // CHECK: %[[SMUL:.*]] = arith.mulf %[[SCALAR1]], %[[SCALAR1]]
  // CHECK: %[[VECTOR1:.*]] = arith.divf %[[CST_V]], %[[ARG1]]
  // CHECK: %[[VMUL:.*]] = arith.mulf %[[VECTOR1]], %[[VECTOR1]]
  // CHECK: return %[[SCALAR0]], %[[VECTOR0]], %[[SMUL]], %[[VMUL]]
  %c1 = arith.constant 2 : i32
  %v1 = arith.constant dense <2> : vector<4xi32>
  %0 = math.fpowi %arg0, %c1 : f32, i32
  %1 = math.fpowi %arg1, %v1 : vector<4xf32>, vector<4xi32>
  %cm1 = arith.constant -2 : i32
  %vm1 = arith.constant dense <-2> : vector<4xi32>
  %2 = math.fpowi %arg0, %cm1 : f32, i32
  %3 = math.fpowi %arg1, %vm1 : vector<4xf32>, vector<4xi32>
  return %0, %1, %2, %3 : f32, vector<4xf32>, f32, vector<4xf32>
}

// CHECK-LABEL: @fpowi_exp_three(
// CHECK-SAME: %[[ARG0:.+]]: f32
// CHECK-SAME: %[[ARG1:.+]]: vector<4xf32>
// CHECK-SAME: -> (f32, vector<4xf32>, f32, vector<4xf32>) {
func.func @fpowi_exp_three(%arg0: f32, %arg1: vector<4xf32>) -> (f32, vector<4xf32>, f32, vector<4xf32>) {
  // CHECK-DAG: %[[CST_S:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[CST_V:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
  // CHECK: %[[SMUL0:.*]] = arith.mulf %[[ARG0]], %[[ARG0]]
  // CHECK: %[[SCALAR0:.*]] = arith.mulf %[[SMUL0]], %[[ARG0]]
  // CHECK: %[[VMUL0:.*]] = arith.mulf %[[ARG1]], %[[ARG1]]
  // CHECK: %[[VECTOR0:.*]] = arith.mulf %[[VMUL0]], %[[ARG1]]
  // CHECK: %[[SCALAR1:.*]] = arith.divf %[[CST_S]], %[[ARG0]]
  // CHECK: %[[SMUL1:.*]] = arith.mulf %[[SCALAR1]], %[[SCALAR1]]
  // CHECK: %[[SMUL2:.*]] = arith.mulf %[[SMUL1]], %[[SCALAR1]]
  // CHECK: %[[VECTOR1:.*]] = arith.divf %[[CST_V]], %[[ARG1]]
  // CHECK: %[[VMUL1:.*]] = arith.mulf %[[VECTOR1]], %[[VECTOR1]]
  // CHECK: %[[VMUL2:.*]] = arith.mulf %[[VMUL1]], %[[VECTOR1]]
  // CHECK: return %[[SCALAR0]], %[[VECTOR0]], %[[SMUL2]], %[[VMUL2]]
  %c1 = arith.constant 3 : i32
  %v1 = arith.constant dense <3> : vector<4xi32>
  %0 = math.fpowi %arg0, %c1 : f32, i32
  %1 = math.fpowi %arg1, %v1 : vector<4xf32>, vector<4xi32>
  %cm1 = arith.constant -3 : i32
  %vm1 = arith.constant dense <-3> : vector<4xi32>
  %2 = math.fpowi %arg0, %cm1 : f32, i32
  %3 = math.fpowi %arg1, %vm1 : vector<4xf32>, vector<4xi32>
  return %0, %1, %2, %3 : f32, vector<4xf32>, f32, vector<4xf32>
}
