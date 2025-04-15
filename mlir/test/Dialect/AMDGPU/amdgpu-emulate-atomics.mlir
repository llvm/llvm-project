// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx90a %s | FileCheck %s --check-prefixes=CHECK,GFX90A
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1030 %s | FileCheck %s --check-prefixes=CHECK,GFX10
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1100 %s | FileCheck %s --check-prefixes=CHECK,GFX11
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1200 %s | FileCheck %s --check-prefixes=CHECK,GFX12
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx942 %s | FileCheck %s --check-prefixes=CHECK,GFX942
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx950 %s | FileCheck %s --check-prefixes=CHECK,GFX950

// -----

func.func @atomic_fmax(%val: f32, %buffer: memref<?xf32>, %idx: i32) {
// CHECK: func @atomic_fmax
// CHECK-SAME: ([[val:%.+]]: f32, [[buffer:%.+]]: memref<?xf32>, [[idx:%.+]]: i32)
// CHECK: gpu.printf "Begin\0A"
// GFX10: amdgpu.raw_buffer_atomic_fmax {foo, indexOffset = 4 : i32} [[val]] -> [[buffer]][[[idx]]]
// GFX11: amdgpu.raw_buffer_atomic_fmax {foo, indexOffset = 4 : i32} [[val]] -> [[buffer]][[[idx]]]
// GFX12: amdgpu.raw_buffer_atomic_fmax {foo, indexOffset = 4 : i32} [[val]] -> [[buffer]][[[idx]]]
// GFX90A:  [[ld:%.+]] = amdgpu.raw_buffer_load {foo, indexOffset = 4 : i32} [[buffer]][[[idx]]]
// GFX90A:  cf.br [[loop:\^.+]]([[ld]] : f32)
// GFX90A:  [[loop]]([[arg:%.+]]: f32):
// GFX90A:  [[operated:%.+]] = arith.maximumf [[val]], [[arg]]
// GFX90A:  [[atomicRes:%.+]] = amdgpu.raw_buffer_atomic_cmpswap {foo, indexOffset = 4 : i32} [[operated]], [[arg]] -> [[buffer]][[[idx]]]
// GFX90A:  [[argCast:%.+]] = arith.bitcast [[arg]] : f32 to i32
// GFX90A:  [[resCast:%.+]] = arith.bitcast [[atomicRes]] : f32 to i32
// GFX90A:  [[test:%.+]] = arith.cmpi eq, [[resCast]], [[argCast]]
// GFX90A:  cf.cond_br [[test]], [[post:\^.+]], [[loop]]([[atomicRes]] : f32)
// GFX90A:  [[post]]:
// GFX942:  [[ld:%.+]] = amdgpu.raw_buffer_load {foo, indexOffset = 4 : i32} [[buffer]][[[idx]]]
// GFX942:  cf.br [[loop:\^.+]]([[ld]] : f32)
// GFX942:  [[loop]]([[arg:%.+]]: f32):
// GFX942:  [[operated:%.+]] = arith.maximumf [[val]], [[arg]]
// GFX942:  [[atomicRes:%.+]] = amdgpu.raw_buffer_atomic_cmpswap {foo, indexOffset = 4 : i32} [[operated]], [[arg]] -> [[buffer]][[[idx]]]
// GFX942:  [[argCast:%.+]] = arith.bitcast [[arg]] : f32 to i32
// GFX942:  [[resCast:%.+]] = arith.bitcast [[atomicRes]] : f32 to i32
// GFX942:  [[test:%.+]] = arith.cmpi eq, [[resCast]], [[argCast]]
// GFX942:  cf.cond_br [[test]], [[post:\^.+]], [[loop]]([[atomicRes]] : f32)
// GFX942:  [[post]]:
// GFX950:  [[ld:%.+]] = amdgpu.raw_buffer_load {foo, indexOffset = 4 : i32} [[buffer]][[[idx]]]
// GFX950:  cf.br [[loop:\^.+]]([[ld]] : f32)
// GFX950:  [[loop]]([[arg:%.+]]: f32):
// GFX950:  [[operated:%.+]] = arith.maximumf [[val]], [[arg]]
// GFX950:  [[atomicRes:%.+]] = amdgpu.raw_buffer_atomic_cmpswap {foo, indexOffset = 4 : i32} [[operated]], [[arg]] -> [[buffer]][[[idx]]]
// GFX950:  [[argCast:%.+]] = arith.bitcast [[arg]] : f32 to i32
// GFX950:  [[resCast:%.+]] = arith.bitcast [[atomicRes]] : f32 to i32
// GFX950:  [[test:%.+]] = arith.cmpi eq, [[resCast]], [[argCast]]
// GFX950:  cf.cond_br [[test]], [[post:\^.+]], [[loop]]([[atomicRes]] : f32)
// GFX950:  [[post]]:
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
// GFX90A:  amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// GFX10: amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// GFX11: amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// GFX12: amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// GFX942: amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// GFX950: amdgpu.raw_buffer_atomic_fmax [[val]] -> [[buffer]][[[idx]]]
// CHECK-NEXT: gpu.printf "End\0A"
  gpu.printf "Begin\n"
  amdgpu.raw_buffer_atomic_fmax %val -> %buffer[%idx] : f64 -> memref<?xf64>, i32
  gpu.printf "End\n"
  func.return
}

// -----

func.func @atomic_fadd(%val: f32, %buffer: memref<?xf32>, %idx: i32) {
// CHECK: func @atomic_fadd
// GFX90A:  amdgpu.raw_buffer_atomic_fadd
// GFX10: amdgpu.raw_buffer_load
// GFX10: amdgpu.raw_buffer_atomic_cmpswap
// GFX11: amdgpu.raw_buffer_atomic_fadd
// GFX12: amdgpu.raw_buffer_atomic_fadd
// GFX942: amdgpu.raw_buffer_atomic_fadd
// GFX950: amdgpu.raw_buffer_atomic_fadd
  amdgpu.raw_buffer_atomic_fadd %val -> %buffer[%idx] : f32 -> memref<?xf32>, i32
  func.return
}

// CHECK: func @atomic_fadd_v2f16
func.func @atomic_fadd_v2f16(%val: vector<2xf16>, %buffer: memref<?xf16>, %idx: i32) {
// GFX90A:  amdgpu.raw_buffer_atomic_fadd
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
// GFX942: amdgpu.raw_buffer_atomic_fadd
// GFX12:  amdgpu.raw_buffer_atomic_fadd
// GFX950:  amdgpu.raw_buffer_atomic_fadd
  amdgpu.raw_buffer_atomic_fadd %val -> %buffer[%idx] : vector<2xf16> -> memref<?xf16>, i32
  func.return
}

// CHECK: func @atomic_fadd_v2bf16
func.func @atomic_fadd_v2bf16(%val: vector<2xbf16>, %buffer: memref<?xbf16>, %idx: i32) {
// GFX90A: amdgpu.raw_buffer_load
// GFX90A: amdgpu.raw_buffer_atomic_cmpswap
// GFX10: amdgpu.raw_buffer_load
// GFX10: amdgpu.raw_buffer_atomic_cmpswap
// GFX11: amdgpu.raw_buffer_load
// GFX11: amdgpu.raw_buffer_atomic_cmpswap
// GFX942: amdgpu.raw_buffer_load
// GFX942: amdgpu.raw_buffer_atomic_cmpswap
// GFX12:  amdgpu.raw_buffer_atomic_fadd
// GFX950:  amdgpu.raw_buffer_atomic_fadd
  amdgpu.raw_buffer_atomic_fadd %val -> %buffer[%idx] : vector<2xbf16> -> memref<?xbf16>, i32
  func.return
}
