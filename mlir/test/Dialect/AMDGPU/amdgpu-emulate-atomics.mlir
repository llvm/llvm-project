// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx908 %s | FileCheck %s --check-prefixes=CHECK,GFX9CAS,GFX908
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx90a %s | FileCheck %s --check-prefixes=CHECK,GFX9CAS,GFX90A
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx90c %s | FileCheck %s --check-prefixes=CHECK,GFX9CAS,GFX90C
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1030 %s | FileCheck %s --check-prefixes=CHECK,GFX10
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1100 %s | FileCheck %s --check-prefixes=CHECK,GFX11
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx1200 %s | FileCheck %s --check-prefixes=CHECK,GFX12
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx942 %s | FileCheck %s --check-prefixes=CHECK,GFX9CAS,GFX942
// RUN: mlir-opt -split-input-file -amdgpu-emulate-atomics=chipset=gfx950 %s | FileCheck %s --check-prefixes=CHECK,GFX9CAS,GFX950

// -----

func.func @atomic_fmax(%val: f32, %buffer: memref<?xf32>, %idx: i32) -> f32 {
// CHECK: func @atomic_fmax
// CHECK-SAME: ([[val:%.+]]: f32, [[buffer:%.+]]: memref<?xf32>, [[idx:%.+]]: i32)
// CHECK: gpu.printf "Begin\0A"
// GFX10: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) {foo} [[val]] -> [[buffer]][[[idx]]]
// GFX11: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) {foo} [[val]] -> [[buffer]][[[idx]]]
// GFX12: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) {foo} [[val]] -> [[buffer]][[[idx]]]
// GFX9CAS: [[ld:%.+]] = amdgpu.raw_buffer_load boundsCheck(true) {foo} [[buffer]][[[idx]]]
// GFX9CAS:  cf.br [[loop:\^.+]]([[ld]] : f32)
// GFX9CAS:  [[loop]]([[arg:%.+]]: f32):
// GFX9CAS:  [[operated:%.+]] = arith.maximumf [[val]], [[arg]]
// GFX9CAS: [[atomicRes:%.+]] = amdgpu.raw_buffer_atomic_cmpswap boundsCheck(true) {foo} [[operated]], [[arg]] -> [[buffer]][[[idx]]]
// GFX9CAS:  [[argCast:%.+]] = arith.bitcast [[arg]] : f32 to i32
// GFX9CAS:  [[resCast:%.+]] = arith.bitcast [[atomicRes]] : f32 to i32
// GFX9CAS:  [[test:%.+]] = arith.cmpi eq, [[resCast]], [[argCast]]
// GFX9CAS:  cf.cond_br [[test]], [[post:\^.+]]([[arg]] : f32), [[loop]]([[atomicRes]] : f32)
// GFX9CAS:  [[post]]([[old:%.+]]: f32):
// CHECK-NEXT: gpu.printf "End\0A"
// CHECK-NEXT: return
  gpu.printf "Begin\n"
  %old = amdgpu.raw_buffer_atomic_fmax boundsCheck(true) {foo} %val -> %buffer[%idx] indexOffset(4) : f32 -> memref<?xf32>, i32
  gpu.printf "End\n"
  func.return %old : f32
}

// -----

func.func @atomic_fmax_f64(%val: f64, %buffer: memref<?xf64>, %idx: i32) {
// CHECK: func @atomic_fmax_f64
// CHECK-SAME: ([[val:%.+]]: f64, [[buffer:%.+]]: memref<?xf64>, [[idx:%.+]]: i32)
// CHECK: gpu.printf "Begin\0A"
// GFX90A:  amdgpu.raw_buffer_atomic_fmax boundsCheck(true) [[val]] -> [[buffer]][[[idx]]]
// GFX10: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) [[val]] -> [[buffer]][[[idx]]]
// GFX11: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) [[val]] -> [[buffer]][[[idx]]]
// GFX12: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) [[val]] -> [[buffer]][[[idx]]]
// GFX942: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) [[val]] -> [[buffer]][[[idx]]]
// GFX950: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) [[val]] -> [[buffer]][[[idx]]]
// gfx908 has no f64 buffer fmin/fmax, so it is emulated.
// GFX908:  [[ld:%.+]] = amdgpu.raw_buffer_load boundsCheck(true) [[buffer]][[[idx]]]
// GFX908:  cf.br [[loop:\^.+]]([[ld]] : f64)
// GFX908:  [[loop]]([[arg:%.+]]: f64):
// GFX908:  [[operated:%.+]] = arith.maximumf [[val]], [[arg]]
// GFX908: [[atomicRes:%.+]] = amdgpu.raw_buffer_atomic_cmpswap boundsCheck(true) [[operated]], [[arg]] -> [[buffer]][[[idx]]]
// GFX908:  [[argCast:%.+]] = arith.bitcast [[arg]] : f64 to i64
// GFX908:  [[resCast:%.+]] = arith.bitcast [[atomicRes]] : f64 to i64
// GFX908:  [[test:%.+]] = arith.cmpi eq, [[resCast]], [[argCast]]
// GFX908:  cf.cond_br [[test]], [[post:\^.+]]([[arg]] : f64), [[loop]]([[atomicRes]] : f64)
// GFX908:  [[post]]([[old:%.+]]: f64):
// gfx90c has none either, but sorts after gfx90a by ISA version.
// GFX90C: amdgpu.raw_buffer_atomic_fmax boundsCheck(true) [[val]] -> [[buffer]][[[idx]]]
// CHECK-NEXT: gpu.printf "End\0A"
  gpu.printf "Begin\n"
  %old = amdgpu.raw_buffer_atomic_fmax boundsCheck(true) %val -> %buffer[%idx] : f64 -> memref<?xf64>, i32
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
// GFX908: amdgpu.raw_buffer_atomic_fadd
// GFX90C: amdgpu.raw_buffer_atomic_fadd
  %old = amdgpu.raw_buffer_atomic_fadd boundsCheck(true) %val -> %buffer[%idx] : f32 -> memref<?xf32>, i32
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
// GFX908: amdgpu.raw_buffer_atomic_fadd
// GFX90C: amdgpu.raw_buffer_atomic_fadd
  %old = amdgpu.raw_buffer_atomic_fadd boundsCheck(true) %val -> %buffer[%idx] : vector<2xf16> -> memref<?xf16>, i32
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
// GFX908: amdgpu.raw_buffer_load
// GFX908: amdgpu.raw_buffer_atomic_cmpswap
// GFX90C: amdgpu.raw_buffer_load
// GFX90C: amdgpu.raw_buffer_atomic_cmpswap
  %old = amdgpu.raw_buffer_atomic_fadd boundsCheck(true) %val -> %buffer[%idx] : vector<2xbf16> -> memref<?xbf16>, i32
  func.return
}
