// RUN: mlir-opt %s --split-input-file -math-legalize-to-f32 --arith-emulate-unsupported-floats="source-types=bf16 target-type=f32" -eliminate-explicit-rounding | FileCheck %s

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
// CHECK-DAG: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK-DAG: [[COS:%.+]] = math.cos [[ABSF]]
// CHECK: [[ADDF:%.+]] = arith.addf [[SIN]], [[COS]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[ADDF]]
// CHECK: return [[TRUNCF]] : vector<32x32x32xbf16>
func.func @bf16_branch_vector(%arg0: vector<32x32x32xbf16>) -> vector<32x32x32xbf16> {
  %0 = math.absf %arg0 : vector<32x32x32xbf16>
	%1 = math.sin %0 : vector<32x32x32xbf16>
	%2 = math.cos %0 : vector<32x32x32xbf16>
	%3 = arith.addf %1, %2 : vector<32x32x32xbf16>
  return %3 : vector<32x32x32xbf16>
}

// CHECK-LABEL: @bf16_fma
// CHECK-SAME: ([[ARG0:%.+]]: vector<32x32x32xbf16>, [[ARG1:%.+]]: vector<32x32x32xbf16>, [[ARG2:%.+]]: vector<32x32x32xbf16>)
// CHECK: [[EXTF0:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF0]]
// CHECK-DAG: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF0:%.+]] = arith.truncf [[SIN]]
// CHECK-DAG: [[FMA:%.+]] = math.fma [[TRUNCF0]], [[ARG1]], [[ARG2]]
// CHECK: [[EXTF1:%.+]] = arith.extf [[FMA]]
// CHECK: [[ADDF:%.+]] = arith.addf [[EXTF1]], [[SIN]]
// CHECK: [[TRUNCF1:%.+]] = arith.truncf [[ADDF]]
// CHECK: return [[TRUNCF1]] : vector<32x32x32xbf16>
func.func @bf16_fma(%arg0: vector<32x32x32xbf16>, %arg1: vector<32x32x32xbf16>, %arg2: vector<32x32x32xbf16>) -> vector<32x32x32xbf16> {
  %0 = math.absf %arg0 : vector<32x32x32xbf16>
	%1 = math.sin %0 : vector<32x32x32xbf16>
  %2 = math.fma %1, %arg1, %arg2 : vector<32x32x32xbf16>
	%3 = arith.addf %2, %1 : vector<32x32x32xbf16>
  return %3 : vector<32x32x32xbf16>
}
