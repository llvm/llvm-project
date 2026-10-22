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

// CHECK-LABEL: @matrices_and_modes
func.func @matrices_and_modes(%a: vector<2xf32>, %h: vector<2xf16>, %m: i32) -> (vector<2xf32>, vector<2xf16>, i32) {
  // CHECK: agx.mma %{{.*}} * %{{.*}} + %{{.*}} : vector<2xf32>, vector<2xf32>
  %0 = agx.mma %a * %a + %a : vector<2xf32>, vector<2xf32>
  // CHECK: agx.mma %{{.*}} * %{{.*}} + %{{.*}} : vector<2xf16>, vector<2xf16>
  %1 = agx.mma %h * %h + %h : vector<2xf16>, vector<2xf16>
  // CHECK: agx.memory_barrier{{$}}
  agx.memory_barrier
  // CHECK: agx.memory_barrier scope threadgroup
  agx.memory_barrier scope threadgroup
  // CHECK: agx.set_fpmode %{{.*}} : i32
  agx.set_fpmode %m : i32
  // CHECK: agx.get_fpmode : i32
  %2 = agx.get_fpmode : i32
  return %0, %1, %2 : vector<2xf32>, vector<2xf16>, i32
}

// CHECK-LABEL: @performance_ops
func.func @performance_ops(%A: memref<?x?xf16, 1>, %C: memref<?x?xf32, 1>,
                           %i: index, %h: vector<2xf16>, %acc: vector<32xf32>,
                           %x: f32) -> (vector<2xf32>, vector<32xf32>, f32, i32, index) {
  // CHECK: agx.mma %{{.*}} * %{{.*}} + %{{.*}} negated : vector<2xf16>, vector<2xf32>
  %c = agx.matrix_load %C[%i, %i] : memref<?x?xf32, 1> -> vector<2xf32>
  %0 = agx.mma %h * %h + %c negated : vector<2xf16>, vector<2xf32>
  // CHECK: agx.matrix_load %{{.*}}[%{{.*}}, %{{.*}}] transpose : memref<?x?xf16, 1> -> vector<2xf16>
  %t = agx.matrix_load %A[%i, %i] transpose : memref<?x?xf16, 1> -> vector<2xf16>
  // CHECK: agx.matrix_store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<2xf32> -> memref<?x?xf32, 1>
  agx.matrix_store %0, %C[%i, %i] : vector<2xf32> -> memref<?x?xf32, 1>
  // CHECK: agx.mma_tile %{{.*}}[%{{.*}}, %{{.*}}] * %{{.*}}[%{{.*}}, %{{.*}}] + %{{.*}} tile(32, 32, 8)
  %1 = agx.mma_tile %A[%i, %i] * %A[%i, %i] + %acc tile(32, 32, 8)
      : memref<?x?xf16, 1>, memref<?x?xf16, 1>, vector<32xf32>
  // CHECK: agx.fma_lane_shift %{{.*}}, %{{.*}}, %{{.*}} shift 3 width 4 : f32
  %2 = agx.fma_lane_shift %x, %x, %x shift 3 width 4 : f32
  // CHECK: agx.block_barrier
  agx.block_barrier
  // CHECK: agx.core_id : i32
  %3 = agx.core_id : i32
  // CHECK: agx.uniform_region -> (index)
  %4 = agx.uniform_region -> (index) {
    %n = gpu.grid_dim x
    agx.yield %n : index
  }
  return %0, %1, %2, %3, %4 : vector<2xf32>, vector<32xf32>, f32, i32, index
}
