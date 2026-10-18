// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: @shuffles
func.func @shuffles(%f: f32, %v: vector<2xf32>, %h: f16, %d: f64, %i: i32) {
  // CHECK: agx.shuffle xor %{{.*}}, %{{.*}} : f32
  %0 = agx.shuffle xor %f, %i : f32
  // CHECK: agx.shuffle down %{{.*}}, %{{.*}} scope quad : vector<2xf32>
  %1 = agx.shuffle down %v, %i scope quad : vector<2xf32>
  // CHECK: agx.shuffle idx %{{.*}}, %{{.*}} : f16
  %2 = agx.shuffle idx %h, %i : f16
  // CHECK: agx.shuffle up %{{.*}}, %{{.*}} : f64
  %3 = agx.shuffle up %d, %i : f64
  return
}

// CHECK-LABEL: @reductions
func.func @reductions(%f: f32, %h: f16, %i: i32, %s: i16) {
  // CHECK: agx.reduce add %{{.*}} : f32
  %0 = agx.reduce add %f : f32
  // CHECK: agx.reduce maxnumf %{{.*}} scope quad : f16
  %1 = agx.reduce maxnumf %h scope quad : f16
  // CHECK: agx.reduce add %{{.*}} prefix : i32
  %2 = agx.reduce add %i prefix : i32
  // CHECK: agx.reduce minsi %{{.*}} : i16
  %3 = agx.reduce minsi %s : i16
  return
}

// CHECK-LABEL: @misc
func.func @misc(%p: i1) -> i32 {
  // CHECK: agx.threadgroup_barrier
  agx.threadgroup_barrier
  // CHECK: agx.ballot %{{.*}} scope quad : i32
  %0 = agx.ballot %p scope quad : i32
  return %0 : i32
}
