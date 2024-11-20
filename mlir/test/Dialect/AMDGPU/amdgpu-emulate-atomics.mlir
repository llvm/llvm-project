// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx90a %s | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1030 %s | FileCheck %s --check-prefixes=CHECK,GFX10
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1100 %s | FileCheck %s --check-prefixes=CHECK,GFX11

// -----

func.func @atomic_fmax(%val: f32, %buffer: memref<?xf32>, %idx: i32) {
// CHECK: func @atomic_fmax
// CHECK-SAME: ([[val:%.+]]: f32, [[buffer:%.+]]: memref<?xf32>, [[idx:%.+]]: i32)
// CHECK: gpu.printf "Begin\0A"
// GFX10: amdgpu.raw_buffer_atomic_fmax {foo, indexOffset = 4 : i32} [[val]] -> [[buffer]][[[idx]]]
// GFX11: amdgpu.raw_buffer_atomic_fmax {foo, indexOffset = 4 : i32} [[val]] -> [[buffer]][[[idx]]]
// GFX9:  [[ld:%.+]] = amdgpu.raw_buffer_load {foo, indexOffset = 4 : i32} [[buffer]][[[idx]]]
// GFX9:  cf.br [[loop:\^.+]]([[ld]] : f32)
// GFX9:  [[loop]]([[arg:%.+]]: f32):
// GFX9:  [[operated:%.+]] = arith.maximumf [[val]], [[arg]]
// GFX9:  [[atomicRes:%.+]] = amdgpu.raw_buffer_atomic_cmpswap {foo, indexOffset = 4 : i32} [[operated]], [[arg]] -> [[buffer]][[[idx]]]
// GFX9:  [[argCast:%.+]] = arith.bitcast [[arg]] : f32 to i32
// GFX9:  [[resCast:%.+]] = arith.bitcast [[atomicRes]] : f32 to i32
// GFX9:  [[test:%.+]] = arith.cmpi eq, [[resCast]], [[argCast]]
// GFX9:  cf.cond_br [[test]], [[post:\^.+]], [[loop]]([[atomicRes]] : f32)
// GFX9:  [[post]]:
// CHECK-NEXT: gpu.printf "End\0A"
  gpu.printf "Begin\n"
  amdgpu.raw_buffer_atomic_fmax {foo, indexOffset = 4 : i32} %val -> %buffer[%idx] : f32 -> memref<?xf32>, i32
  gpu.printf "End\n"
  func.return
}

// -----

func.func @atomic_fmax_f64(%val: f64, %buffer: memref<?xf64>, %idx: i32) {
// CHECK: func @atomic_fmax_f64
// CHECK-SAME: ([[val:%.+]]: f64, [[buffer:%.+]]: memref<?xf64>, [[idx:%.+]]: i32)
// CHECK: gpu.printf "Begin\0A"
// GFX9:  amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// GFX10: amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// GFX11: amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// CHECK-NEXT: gpu.printf "End\0A"
  gpu.printf "Begin\n"
  amdgpu.raw_buffer_atomic_fmax %val -> %buffer[%idx] : f64 -> memref<?xf64>, i32
  gpu.printf "End\n"
  func.return
}

// -----

func.func @atomic_fadd(%val: f32, %buffer: memref<?xf32>, %idx: i32) {
// CHECK: func @atomic_fadd
// GFX9:  amdgpu.raw_buffer_atomic_fadd
// GFX10: amdgpu.raw_buffer_load
// GFX10: amdgpu.raw_buffer_atomic_cmpswap
// GFX11: amdgpu.raw_buffer_atomic_fadd
  amdgpu.raw_buffer_atomic_fadd %val -> %buffer[%idx] : f32 -> memref<?xf32>, i32
  func.return
}

// CHECK: func @atomic_fadd_v2f16
func.func @atomic_fadd_v2f16(%val: vector<2xf16>, %buffer: memref<?xf16>, %idx: i32) {
// GFX9:  amdgpu.raw_buffer_atomic_fadd
// GFX10: amdgpu.raw_buffer_load
// GFX10: amdgpu.raw_buffer_atomic_cmpswap
// Note: the atomic operation itself will be done over i32, and then we use bitcasts
// to scalars in order to test for exact bitwise equality instead of float
// equality.
// GFX11: %[[old:.+]] = amdgpu.raw_buffer_atomic_cmpswap
// GFX11: %[[vecCastExpected:.+]] = vector.bitcast %{{.*}} : vector<2xf16> to vector<1xi32>
// GFX11: %[[scalarExpected:.+]] = vector.extract %[[vecCastExpected]][0]
// GFX11: %[[vecCastOld:.+]] = vector.bitcast %[[old]] : vector<2xf16> to vector<1xi32>
// GFX11: %[[scalarOld:.+]] = vector.extract %[[vecCastOld]][0]
// GFX11: arith.cmpi eq, %[[scalarOld]], %[[scalarExpected]]
  amdgpu.raw_buffer_atomic_fadd %val -> %buffer[%idx] : vector<2xf16> -> memref<?xf16>, i32
  func.return
}
