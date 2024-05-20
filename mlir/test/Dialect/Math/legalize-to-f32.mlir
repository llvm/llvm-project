// RUN: mlir-opt %s --split-input-file -math-legalize-to-f32=use-canonicalize-f32-promotion=true | FileCheck %s

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
// CHECK-SAME: ([[ARG0:%.+]]: bf16)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : bf16
func.func @sequences(%arg0: bf16) -> bf16 {
  %0 = math.absf %arg0 : bf16
  %1 = math.sin %0 : bf16
  return %1 : bf16
}

// CHECK-LABEL: @eliminatecastoncastf16
// CHECK: return [[arg0:%.+]] : f32
func.func @eliminatecastoncastf16(%arg0: f32) -> f32 {
  %0 = arith.truncf %arg0 : f32 to f16
  %1 = arith.extf %0 : f16 to f32
  return %1 : f32
}

// CHECK-LABEL: @eliminatecastoncastbf16
// CHECK: return [[arg0:%.+]] : f32
func.func @eliminatecastoncastbf16(%arg0: f32) -> f32 {
  %0 = arith.truncf %arg0 : f32 to bf16
  %1 = arith.extf %0 : bf16 to f32
  return %1 : f32
}

// CHECK-LABEL: @bf16_sin_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<32x32x32xbf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : vector<32x32x32xbf16>
func.func @bf16_sin_vector(%arg0: vector<32x32x32xbf16>) -> vector<32x32x32xbf16> {
  %0 = math.absf %arg0 : vector<32x32x32xbf16>
  %1 = math.sin %0 : vector<32x32x32xbf16>
  return %1 : vector<32x32x32xbf16>
}

// CHECK-LABEL: @f16_sin_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<32x32x32xf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : vector<32x32x32xf16>
func.func @f16_sin_vector(%arg0: vector<32x32x32xf16>) -> vector<32x32x32xf16> {
  %0 = math.absf %arg0 : vector<32x32x32xf16>
  %1 = math.sin %0 : vector<32x32x32xf16>
  return %1 : vector<32x32x32xf16>
}

// CHECK-LABEL: @bf16_branch_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<32x32x32xbf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF0:%.+]] = arith.truncf [[SIN]]
// CHECK: [[COS:%.+]] = math.cos [[ABSF]]
// CHECK: [[TRUNCF1:%.+]] = arith.truncf [[COS]]
// CHECK: [[ADDF:%.+]] = arith.addf
// CHECK: return [[ADDF]] : vector<32x32x32xbf16>
func.func @bf16_branch_vector(%arg0: vector<32x32x32xbf16>) -> vector<32x32x32xbf16> {
  %0 = math.absf %arg0 : vector<32x32x32xbf16>
	%1 = math.sin %0 : vector<32x32x32xbf16>
	%2 = math.cos %0 : vector<32x32x32xbf16>
	%3 = arith.addf %1, %2 : vector<32x32x32xbf16>
  return %3 : vector<32x32x32xbf16>
}
