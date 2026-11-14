// RUN: mlir-opt %s -math-expand-ops | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-opt %s -math-expand-ops=expand-round-f64=false | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-opt %s -math-expand-ops="ops=roundeven expand-round-f64=true" | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-opt %s -math-expand-ops="ops=round expand-round-f64=true" | FileCheck %s --check-prefix=ROUND-ONLY
// RUN: mlir-opt %s -math-expand-ops=expand-round-f64=true | FileCheck %s --check-prefix=CHECK --enable-var-scope --implicit-check-not=scf.if --implicit-check-not=math.round
// RUN: mlir-opt %s -math-expand-ops=expand-round-f64=true -inline -canonicalize | FileCheck %s --check-prefix=VALUES --enable-var-scope
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(math-expand-ops{expand-round-f64=true},convert-math-to-llvm,convert-arith-to-llvm),convert-vector-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | FileCheck %s --check-prefix=LLVM --enable-var-scope --implicit-check-not=scf.if --implicit-check-not=llvm.cond_br

// DEFAULT-LABEL: func.func @round_f64(
// DEFAULT: math.round
// ROUND-ONLY-LABEL: func.func @round_f64(
// ROUND-ONLY: arith.shli
// ROUND-ONLY-NOT: math.round
// ROUND-ONLY: return
// CHECK-LABEL: func.func @round_f64(
// CHECK-DAG: %[[MASK:.*]] = arith.constant 63 : i64
// CHECK: %[[SHIFT:.*]] = arith.andi %{{.*}}, %[[MASK]] : i64
// CHECK: arith.shli %{{.*}}, %[[SHIFT]] : i64
// CHECK: arith.select
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// LLVM-LABEL: llvm.func @round_f64(
// LLVM: llvm.shl
// LLVM: llvm.select
func.func @round_f64(%value: f64) -> f64 {
  %result = math.round %value : f64
  return %result : f64
}

// DEFAULT-LABEL: func.func @round_vector_f64(
// DEFAULT: math.round
// ROUND-ONLY-LABEL: func.func @round_vector_f64(
// ROUND-ONLY: arith.shli
// ROUND-ONLY-NOT: math.round
// ROUND-ONLY: return
// CHECK-LABEL: func.func @round_vector_f64(
// CHECK-DAG: %[[MASK:.*]] = arith.constant dense<63> : vector<2xi64>
// CHECK: %[[SHIFT:.*]] = arith.andi %{{.*}}, %[[MASK]] : vector<2xi64>
// CHECK: arith.shli %{{.*}}, %[[SHIFT]] : vector<2xi64>
// LLVM-LABEL: llvm.func @round_vector_f64(
// LLVM: llvm.shl
// LLVM: llvm.select
func.func @round_vector_f64(%value: vector<2xf64>) -> vector<2xf64> {
  %result = math.round %value : vector<2xf64>
  return %result : vector<2xf64>
}

// DEFAULT-LABEL: func.func @round_scalable_f64(
// DEFAULT: math.round
// ROUND-ONLY-LABEL: func.func @round_scalable_f64(
// ROUND-ONLY: arith.shli
// ROUND-ONLY-NOT: math.round
// ROUND-ONLY: return
// CHECK-LABEL: func.func @round_scalable_f64(
// CHECK-DAG: %[[MASK:.*]] = arith.constant dense<63> : vector<[2]xi64>
// CHECK: %[[SHIFT:.*]] = arith.andi %{{.*}}, %[[MASK]] : vector<[2]xi64>
// CHECK: arith.shli %{{.*}}, %[[SHIFT]] : vector<[2]xi64>
// LLVM-LABEL: llvm.func @round_scalable_f64(
// LLVM: llvm.shl
// LLVM: llvm.select
func.func @round_scalable_f64(%value: vector<[2]xf64>) -> vector<[2]xf64> {
  %result = math.round %value : vector<[2]xf64>
  return %result : vector<[2]xf64>
}

// LLVM-LABEL: llvm.func @masked_f64(
// LLVM-SAME: %[[CHOOSE:[^ :]+]]: i1
// LLVM: llvm.select %[[CHOOSE]],
// LLVM: llvm.return
func.func @masked_f64(%choose: i1, %value: f64) -> f64 {
  %rounded = math.round %value : f64
  %zero = arith.constant 0.0 : f64
  %result = arith.select %choose, %rounded, %zero : f64
  return %result : f64
}

// LLVM-LABEL: llvm.func @masked_vector_f64(
// LLVM-SAME: %[[CHOOSE:[^ :]+]]: i1
// LLVM: llvm.select %[[CHOOSE]],
// LLVM: llvm.return
func.func @masked_vector_f64(%choose: i1, %value: vector<2xf64>) -> vector<2xf64> {
  %rounded = math.round %value : vector<2xf64>
  %zero = arith.constant dense<0.0> : vector<2xf64>
  %result = arith.select %choose, %rounded, %zero : vector<2xf64>
  return %result : vector<2xf64>
}

// LLVM-LABEL: llvm.func @masked_scalable_f64(
// LLVM-SAME: %[[CHOOSE:[^ :]+]]: i1
// LLVM: llvm.select %[[CHOOSE]],
// LLVM: llvm.return
func.func @masked_scalable_f64(%choose: i1, %value: vector<[2]xf64>) -> vector<[2]xf64> {
  %rounded = math.round %value : vector<[2]xf64>
  %zero = arith.constant dense<0.0> : vector<[2]xf64>
  %result = arith.select %choose, %rounded, %zero : vector<[2]xf64>
  return %result : vector<[2]xf64>
}

// VALUES-LABEL: func.func @half_neighbors(
// VALUES: %[[EXPECTED:.*]] = arith.constant dense<[0, 4607182418800017408]> : vector<2xi64>
// VALUES-NEXT: return %[[EXPECTED]] : vector<2xi64>
func.func @half_neighbors() -> vector<2xi64> {
  %input = arith.constant dense<[4602678819172646911, 4602678819172646912]> : vector<2xi64>
  %value = arith.bitcast %input : vector<2xi64> to vector<2xf64>
  %rounded = func.call @round_vector_f64(%value) : (vector<2xf64>) -> vector<2xf64>
  %result = arith.bitcast %rounded : vector<2xf64> to vector<2xi64>
  return %result : vector<2xi64>
}

// VALUES-LABEL: func.func @ties(
// VALUES: %[[EXPECTED:.*]] = arith.constant dense<[4611686018427387904, -4609434218613702656]> : vector<2xi64>
// VALUES-NEXT: return %[[EXPECTED]] : vector<2xi64>
func.func @ties() -> vector<2xi64> {
  %input = arith.constant dense<[4609434218613702656, -4610560118520545280]> : vector<2xi64>
  %value = arith.bitcast %input : vector<2xi64> to vector<2xf64>
  %rounded = func.call @round_vector_f64(%value) : (vector<2xf64>) -> vector<2xf64>
  %result = arith.bitcast %rounded : vector<2xf64> to vector<2xi64>
  return %result : vector<2xi64>
}

// VALUES-LABEL: func.func @signed_subnormals(
// VALUES: %[[EXPECTED:.*]] = arith.constant dense<[0, -9223372036854775808]> : vector<2xi64>
// VALUES-NEXT: return %[[EXPECTED]] : vector<2xi64>
func.func @signed_subnormals() -> vector<2xi64> {
  %input = arith.constant dense<[1, -9223372036854775807]> : vector<2xi64>
  %value = arith.bitcast %input : vector<2xi64> to vector<2xf64>
  %rounded = func.call @round_vector_f64(%value) : (vector<2xf64>) -> vector<2xf64>
  %result = arith.bitcast %rounded : vector<2xf64> to vector<2xi64>
  return %result : vector<2xi64>
}

// VALUES-LABEL: func.func @large_boundary(
// VALUES: %[[EXPECTED:.*]] = arith.constant dense<[4841369599423283200, 4841369599423283201]> : vector<2xi64>
// VALUES-NEXT: return %[[EXPECTED]] : vector<2xi64>
func.func @large_boundary() -> vector<2xi64> {
  %input = arith.constant dense<[4841369599423283199, 4841369599423283201]> : vector<2xi64>
  %value = arith.bitcast %input : vector<2xi64> to vector<2xf64>
  %rounded = func.call @round_vector_f64(%value) : (vector<2xf64>) -> vector<2xf64>
  %result = arith.bitcast %rounded : vector<2xf64> to vector<2xi64>
  return %result : vector<2xi64>
}

// VALUES-LABEL: func.func @special_values(
// VALUES: %[[EXPECTED:.*]] = arith.constant dense<[9218868437227405312, 9221120237041095220]> : vector<2xi64>
// VALUES-NEXT: return %[[EXPECTED]] : vector<2xi64>
func.func @special_values() -> vector<2xi64> {
  %input = arith.constant dense<[9218868437227405312, 9221120237041095220]> : vector<2xi64>
  %value = arith.bitcast %input : vector<2xi64> to vector<2xf64>
  %rounded = func.call @round_vector_f64(%value) : (vector<2xf64>) -> vector<2xf64>
  %result = arith.bitcast %rounded : vector<2xf64> to vector<2xi64>
  return %result : vector<2xi64>
}

// VALUES-LABEL: func.func @negative_zero(
// VALUES: %[[EXPECTED:.*]] = arith.constant -9223372036854775808 : i64
// VALUES-NEXT: return %[[EXPECTED]] : i64
func.func @negative_zero() -> i64 {
  %input = arith.constant -9223372036854775808 : i64
  %value = arith.bitcast %input : i64 to f64
  %rounded = func.call @round_f64(%value) : (f64) -> f64
  %result = arith.bitcast %rounded : f64 to i64
  return %result : i64
}

// VALUES-LABEL: func.func @negative_infinity(
// VALUES: %[[EXPECTED:.*]] = arith.constant -4503599627370496 : i64
// VALUES-NEXT: return %[[EXPECTED]] : i64
func.func @negative_infinity() -> i64 {
  %input = arith.constant -4503599627370496 : i64
  %value = arith.bitcast %input : i64 to f64
  %rounded = func.call @round_f64(%value) : (f64) -> f64
  %result = arith.bitcast %rounded : f64 to i64
  return %result : i64
}

// VALUES-LABEL: func.func @signaling_nan(
// VALUES: %[[EXPECTED:.*]] = arith.constant 9218868437227405313 : i64
// VALUES-NEXT: return %[[EXPECTED]] : i64
func.func @signaling_nan() -> i64 {
  %input = arith.constant 9218868437227405313 : i64
  %value = arith.bitcast %input : i64 to f64
  %rounded = func.call @round_f64(%value) : (f64) -> f64
  %result = arith.bitcast %rounded : f64 to i64
  return %result : i64
}

// DEFAULT-LABEL: func.func @roundeven_f64(
// ROUND-ONLY-LABEL: func.func @roundeven_f64(
// ROUND-ONLY: math.roundeven
// DEFAULT: math.round
// DEFAULT-NOT: math.roundeven
// DEFAULT: return
// CHECK-LABEL: func.func @roundeven_f64(
// CHECK: arith.shli
// CHECK: return
// LLVM-LABEL: llvm.func @roundeven_f64(
// LLVM: llvm.shl
// LLVM: llvm.return
func.func @roundeven_f64(%value: f64) -> f64 {
  %result = math.roundeven %value : f64
  return %result : f64
}

// VALUES-LABEL: func.func @even_tie(
// VALUES: %[[EXPECTED:.*]] = arith.constant 4611686018427387904 : i64
// VALUES-NEXT: return %[[EXPECTED]] : i64
func.func @even_tie() -> i64 {
  %value = arith.constant 2.5 : f64
  %rounded = func.call @roundeven_f64(%value) : (f64) -> f64
  %result = arith.bitcast %rounded : f64 to i64
  return %result : i64
}
