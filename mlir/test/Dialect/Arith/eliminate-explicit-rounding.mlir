// RUN: mlir-opt %s --split-input-file --arith-eliminate-explicit-rounding | FileCheck %s

// CHECK-LABEL: @sequences
// CHECK-SAME: ([[ARG0:%.+]]: bf16)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : bf16
func.func @sequences(%arg0: bf16) -> bf16 {
  %0 = arith.extf %arg0 : bf16 to f32
  %1 = math.absf %0 : f32
  %2 = arith.truncf %1 : f32 to bf16
  %3 = arith.extf %2 : bf16 to f32
  %4 = math.sin %3 : f32
  %5 = arith.truncf %4 : f32 to bf16
  return %5 : bf16
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
  %0 = arith.extf %arg0 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %1 = math.absf %0 : vector<32x32x32xf32>
  %2 = arith.truncf %1 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  %3 = arith.extf %2 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %4 = math.sin %3 : vector<32x32x32xf32>
  %5 = arith.truncf %4 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  return %5 : vector<32x32x32xbf16>
}

// CHECK-LABEL: @f16_sin_vector
// CHECK-SAME: ([[ARG0:%.+]]: vector<32x32x32xf16>)
// CHECK: [[EXTF:%.+]] = arith.extf [[ARG0]]
// CHECK: [[ABSF:%.+]] = math.absf [[EXTF]]
// CHECK: [[SIN:%.+]] = math.sin [[ABSF]]
// CHECK: [[TRUNCF:%.+]] = arith.truncf [[SIN]]
// CHECK: return [[TRUNCF]] : vector<32x32x32xf16>
func.func @f16_sin_vector(%arg0: vector<32x32x32xf16>) -> vector<32x32x32xf16> {
  %0 = arith.extf %arg0 : vector<32x32x32xf16> to vector<32x32x32xf32>
  %1 = math.absf %0 : vector<32x32x32xf32>
  %2 = arith.truncf %1 : vector<32x32x32xf32> to vector<32x32x32xf16>
  %3 = arith.extf %2 : vector<32x32x32xf16> to vector<32x32x32xf32>
  %4 = math.sin %3 : vector<32x32x32xf32>
  %5 = arith.truncf %4 : vector<32x32x32xf32> to vector<32x32x32xf16>
  return %5 : vector<32x32x32xf16>
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
  %0 = arith.extf %arg0 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %1 = math.absf %0 : vector<32x32x32xf32>
  %2 = arith.truncf %1 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  %3 = arith.extf %2 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %4 = math.sin %3 : vector<32x32x32xf32>
  %5 = arith.truncf %4 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  %6 = arith.extf %5 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %7 = math.cos %3 : vector<32x32x32xf32>
  %8 = arith.truncf %7 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  %9 = arith.extf %8 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %10 = arith.addf %6, %9 : vector<32x32x32xf32>
  %11 = arith.truncf %10 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  return %11 : vector<32x32x32xbf16>
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
  %0 = arith.extf %arg0 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %1 = math.absf %0 : vector<32x32x32xf32>
  %2 = arith.truncf %1 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  %3 = arith.extf %2 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %4 = math.sin %3 : vector<32x32x32xf32>
  %5 = arith.truncf %4 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  %6 = arith.extf %5 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %7 = math.fma %5, %arg1, %arg2 : vector<32x32x32xbf16>
  %8 = arith.extf %7 : vector<32x32x32xbf16> to vector<32x32x32xf32>
  %9 = arith.addf %8, %6 : vector<32x32x32xf32>
  %10 = arith.truncf %9 : vector<32x32x32xf32> to vector<32x32x32xbf16>
  return %10 : vector<32x32x32xbf16>
}
