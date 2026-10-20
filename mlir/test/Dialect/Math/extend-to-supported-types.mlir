// RUN: mlir-opt %s --split-input-file -math-extend-to-supported-types="target-type=f32" | FileCheck %s

// CHECK-LABEL: @sin
// CHECK-SAME: ([[ARG0:%.+]]: f16)
func.func @sin(%arg0: f16) -> f16 {
  // CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
  // CHECK: [[SIN:%.+]] = math.sin [[EXTF]]
  // CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
  // CHECK: return [[TRUNCF]] : f16
  %0 = math.sin %arg0 : f16
  return %0 : f16
}

// CHECK-LABEL: @fpowi
// CHECK-SAME: ([[ARG0:%.+]]: f16, [[ARG1:%.+]]: i32)
func.func @fpowi(%arg0: f16, %arg1: i32) -> f16 {
  // CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
  // CHECK: [[FPOWI:%.+]] = math.fpowi [[EXTF]], [[ARG1]]
  // CHECK: [[TRUNCF:%.+]] = arith.truncf [[FPOWI]]
  // CHECK: return [[TRUNCF]] : f16
  %0 = math.fpowi %arg0, %arg1 : f16, i32
  return %0 : f16
}

// COM: Verify that the pass leaves `math.fma` untouched, since it is often
// COM: implemented on small data types.
// CHECK-LABEL: @fma
// CHECK-SAME: ([[ARG0:%.+]]: f16, [[ARG1:%.+]]: f16, [[ARG2:%.+]]: f16)
// CHECK: [[FMA:%.+]] = math.fma [[ARG0]], [[ARG1]], [[ARG2]]
// CHECK: return [[FMA]] : f16
func.func @fma(%arg0: f16, %arg1: f16, %arg2: f16) -> f16 {
  %0 = math.fma %arg0, %arg1, %arg2 : f16
  return %0 : f16
}

// CHECK-LABEL: @absf_f32
// CHECK-SAME: ([[ARG0:%.+]]: f32)
// CHECK: [[ABSF:%.+]] = math.absf [[ARG0]]
// CHECK: return [[ABSF]] : f32
func.func @absf_f32(%arg0: f32) -> f32 {
  %0 = math.absf %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @absf_f64
// CHECK-SAME: ([[ARG0:%.+]]: f64)
// CHECK: [[ABSF:%.+]] = math.absf [[ARG0]]
// CHECK: return [[ABSF]] : f64
func.func @absf_f64(%arg0: f64) -> f64 {
  %0 = math.absf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: @sin_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<2xbf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[SIN:%.+]] = math.sin [[EXTF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : vector<2xbf16>
func.func @sin_vector(%arg0: vector<2xbf16>) -> vector<2xbf16> {
  %0 = math.sin %arg0 : vector<2xbf16>
  return %0 : vector<2xbf16>
}

// CHECK-LABEL: @fastmath
// CHECK: math.sin %{{.+}} fastmath<nsz>
func.func @fastmath(%arg0: f16) -> f16 {
  %0 = math.sin %arg0 fastmath<nsz> : f16
  return %0 : f16
}

// CHECK-LABEL: @sequences
// CHECK-SAME: ([[ARG0:%.+]]: f16)
// CHECK: [[EXTF0:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF0]]
// CHECK: [[TRUNCF0:%.+]] = arith.truncf [[ABSF]]
// CHECK: [[EXTF1:%.+]] = arith.extf [[TRUNCF0]]
// CHECK: [[SIN:%.+]] = math.sin [[EXTF1]]
// CHECK: [[TRUNCF1:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF1]] : f16
func.func @sequences(%arg0: f16) -> f16 {
  %0 = math.absf %arg0 : f16
  %1 = math.sin %0 : f16
  return %1 : f16
}

// CHECK-LABEL: @promote_in_if_block
func.func @promote_in_if_block(%arg0: bf16, %arg1: bf16, %arg2: i1) -> bf16 {
  // CHECK: [[EXTF0:%.+]] = arith.extf
  // CHECK-NEXT: %[[RES:.*]] = scf.if
  %0 = scf.if %arg2 -> bf16 {
    %1 = math.absf %arg0 : bf16
    // CHECK: [[TRUNCF0:%.+]] = arith.truncf
    scf.yield %1 : bf16
  } else {
    scf.yield %arg1 : bf16
  }
  return %0 : bf16
}
