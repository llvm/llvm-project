// RUN: mlir-opt %s -agx-expand-ops=chip=agx-g14c -split-input-file | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: mlir-opt %s -agx-expand-ops="chip=agx-g14c features=+mesa-isa" -split-input-file | FileCheck %s --check-prefixes=CHECK,PORTABLE
// RUN: mlir-opt %s -agx-expand-ops=chip=agx-g13g -split-input-file | FileCheck %s --check-prefixes=CHECK,PORTABLE

// A fragment is loaded with two loads per thread at the documented layout.
// CHECK-LABEL: @load
// CHECK: gpu.lane_id
// CHECK-COUNT-2: memref.load
// CHECK: vector.from_elements
func.func @load(%A: memref<?x?xf32>, %i: index) -> vector<2xf32> {
  %0 = agx.matrix_load %A[%i, %i] : memref<?x?xf32> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// -----

// A 16x16x16 tile: 2 k steps, each loading 2 A and 2 B fragments and doing
// 4 same-type products (the f16 inputs are extended for the f32 accumulator).
// CHECK-LABEL: @tile
// CHECK-NOT: agx.mma_tile
// CHECK-COUNT-8: memref.load
// CHECK-COUNT-4: agx.mma %{{.*}} * %{{.*}} + %{{.*}} : vector<2xf32>, vector<2xf32>
// CHECK-COUNT-8: memref.load
// CHECK-COUNT-4: agx.mma
// CHECK-NOT: agx.mma
func.func @tile(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %acc: vector<8xf32>, %i: index) -> vector<8xf32> {
  %0 = agx.mma_tile %A[%i, %i] * %B[%i, %i] + %acc tile(16, 16, 16)
      : memref<?x?xf16>, memref<?x?xf16>, vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// G14 has no native negated accumulator.
// CHECK-LABEL: @negate
// CHECK: arith.negf
// CHECK: agx.mma %{{.*}} * %{{.*}} + %{{.*}} : vector<2xf32>, vector<2xf32>
// CHECK-NOT: negated
func.func @negate(%a: vector<2xf32>, %c: vector<2xf32>) -> vector<2xf32> {
  %0 = agx.mma %a * %a + %c negated : vector<2xf32>, vector<2xf32>
  return %0 : vector<2xf32>
}

// -----

// CHECK-LABEL: @lane_shift
// CHECK: agx.shuffle up
// CHECK: agx.ballot
// CHECK: arith.select
// CHECK: math.fma
func.func @lane_shift(%a: f32, %b: f32, %c: f32) -> f32 {
  %0 = agx.fma_lane_shift %a, %b, %c shift 1 width 16 : f32
  return %0 : f32
}

// -----

// G14 has the block barrier; G13 and Mesa ISA mode use the threadgroup one.
// CHECK-LABEL: @barrier_and_uniform
// NATIVE: agx.block_barrier
// PORTABLE: agx.threadgroup_barrier
// CHECK-NOT: agx.uniform_region
// CHECK: gpu.grid_dim x
func.func @barrier_and_uniform() -> index {
  agx.block_barrier
  %0 = agx.uniform_region -> (index) {
    %n = gpu.grid_dim x
    agx.yield %n : index
  }
  return %0 : index
}
