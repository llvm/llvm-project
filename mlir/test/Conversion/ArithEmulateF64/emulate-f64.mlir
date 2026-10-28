// RUN: mlir-opt %s --arith-emulate-f64 --split-input-file | FileCheck %s

// CHECK: llvm.func internal @__air_f64_add

// CHECK-LABEL: func.func @add
// CHECK-SAME: (%[[A:.*]]: i64, %[[B:.*]]: i64) -> i64
// CHECK: %[[R:.*]] = llvm.call @__air_f64_add(%[[A]], %[[B]])
// CHECK: return %[[R]] : i64
// CHECK-NOT: {{[ <x]}}f64{{[ >,)]}}
func.func @add(%a: f64, %b: f64) -> f64 {
  %r = arith.addf %a, %b : f64
  return %r : f64
}

// -----

// CHECK-LABEL: func.func @ops
// CHECK: llvm.call @__air_f64_mul
// CHECK: llvm.call @__air_f64_div
// CHECK: llvm.call @__air_f64_fma
// CHECK: llvm.call @__air_f64_sqrt
// CHECK-NOT: {{[ <x]}}f64{{[ >,)]}}
func.func @ops(%a: f64, %b: f64) -> f64 {
  %m = arith.mulf %a, %b : f64
  %d = arith.divf %m, %b : f64
  %f = math.fma %a, %b, %d : f64
  %s = math.sqrt %f : f64
  return %s : f64
}

// -----

// CHECK-LABEL: func.func @neg_abs
// CHECK: arith.xori
// CHECK: arith.andi
func.func @neg_abs(%a: f64) -> f64 {
  %n = arith.negf %a : f64
  %b = math.absf %n : f64
  return %b : f64
}

// -----

// CHECK-LABEL: func.func @cmp
// CHECK: llvm.call @__air_f64_lt
func.func @cmp(%a: f64, %b: f64) -> i1 {
  %c = arith.cmpf olt, %a, %b : f64
  return %c : i1
}

// -----

// CHECK-LABEL: func.func @conv
// CHECK: llvm.call @__air_f64_from_f32
// CHECK: llvm.call @__air_f64_to_i64
// CHECK: llvm.call @__air_f64_to_f32
func.func @conv(%a: f32) -> (i64, f32) {
  %d = arith.extf %a : f32 to f64
  %i = arith.fptosi %d : f64 to i64
  %t = arith.truncf %d : f64 to f32
  return %i, %t : i64, f32
}

// -----

// CHECK-LABEL: func.func @mem
// CHECK-SAME: memref<4xi64>
// CHECK: memref.load {{.*}} : memref<4xi64>
// CHECK: memref.store {{.*}} : memref<4xi64>
func.func @mem(%m: memref<4xf64>, %i: index) {
  %v = memref.load %m[%i] : memref<4xf64>
  %c = arith.constant 1.0 : f64
  %r = arith.addf %v, %c : f64
  memref.store %r, %m[%i] : memref<4xf64>
  return
}

// -----

// CHECK-LABEL: func.func @atomic
// CHECK: scf.while
// CHECK: llvm.cmpxchg
func.func @atomic(%m: memref<4xf64>, %i: index, %v: f64) -> f64 {
  %r = memref.atomic_rmw addf %v, %m[%i] : (f64, memref<4xf64>) -> f64
  return %r : f64
}

// -----

// CHECK-LABEL: func.func @loop
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i64)
func.func @loop(%n: index, %x: f64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %x) -> (f64) {
    %b = arith.mulf %a, %a : f64
    scf.yield %b : f64
  }
  return %r : f64
}

// -----

// A SIMD-group sum of f64: five shuffles of the bit patterns, each added by the
// library (not an integer reduction of the bit patterns).
// CHECK-LABEL: func.func @subgroup_sum
// CHECK-NOT: gpu.subgroup_reduce
// CHECK-COUNT-5: gpu.shuffle xor {{.*}} : i64
// CHECK-NOT: gpu.subgroup_reduce
// CHECK: llvm.call @__air_f64_add
func.func @subgroup_sum(%x: f64) -> f64 {
  %r = gpu.subgroup_reduce add %x : (f64) -> f64
  return %r : f64
}

// -----

// f32 reductions stay for the target's own lowering.
// CHECK-LABEL: func.func @subgroup_sum_f32
// CHECK: gpu.subgroup_reduce add %{{.*}} : (f32) -> f32
func.func @subgroup_sum_f32(%x: f32) -> f32 {
  %r = gpu.subgroup_reduce add %x : (f32) -> f32
  return %r : f32
}
