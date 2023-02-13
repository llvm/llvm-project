// RUN: mlir-opt %s \
// RUN: | mlir-opt -convert-scf-to-cf \
// RUN: | mlir-opt -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl{chipset=%chip index-bitwidth=32},gpu-to-hsaco{chip=%chip}))' \
// RUN: | mlir-opt -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_rocm_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// TODO: swap for vector transfer reads if we ever create a --vector-to-amdgpu
func.func @vectransferx2(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) {
  %cst = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst, %block_z = %cst) {
    %f0 = arith.constant 0.0: f32
    %base = arith.constant 0 : i32
    %f = amdgpu.raw_buffer_load {boundsCheck = true } %arg0[%base]
      : memref<?xf32>, i32 -> vector<2xf32>

    %c = arith.addf %f, %f : vector<2xf32>

    %base1 = arith.constant 1 : i32
    amdgpu.raw_buffer_store { boundsCheck = false } %c -> %arg1[%base1]
      : vector<2xf32> -> memref<?xf32>, i32

    gpu.terminator
  }
  return
}

func.func @vectransferx4(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) {
  %cst = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst, %block_z = %cst) {
    %f0 = arith.constant 0.0: f32
    %base = arith.constant 0 : i32
    %f = amdgpu.raw_buffer_load { boundsCheck = false } %arg0[%base]
      : memref<?xf32>, i32 -> vector<4xf32>

    %c = arith.addf %f, %f : vector<4xf32>

    amdgpu.raw_buffer_store { boundsCheck = false } %c -> %arg1[%base]
      : vector<4xf32> -> memref<?xf32>, i32

    gpu.terminator
  }
  return
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf1 = arith.constant 1.0 : f32
  %cf1dot23 = arith.constant 1.23 : f32

  %arg0 = memref.alloc() : memref<4xf32>
  %arg1 = memref.alloc() : memref<4xf32>

  %22 = memref.cast %arg0 : memref<4xf32> to memref<?xf32>
  %23 = memref.cast %arg1 : memref<4xf32> to memref<?xf32>

  scf.for %i = %c0 to %c4 step %c1 {
    memref.store %cf1dot23, %22[%i] : memref<?xf32>
    memref.store %cf1dot23, %23[%i] : memref<?xf32>
  }

  %cast0 = memref.cast %22 : memref<?xf32> to memref<*xf32>
  %cast1 = memref.cast %23 : memref<?xf32> to memref<*xf32>

  gpu.host_register %cast0 : memref<*xf32>
  gpu.host_register %cast1 : memref<*xf32>

  %24 = call @mgpuMemGetDeviceMemRef1dFloat(%22) : (memref<?xf32>) -> (memref<?xf32>)
  %26 = call @mgpuMemGetDeviceMemRef1dFloat(%23) : (memref<?xf32>) -> (memref<?xf32>)

  // CHECK: [1.23, 2.46, 2.46, 1.23]
  call @vectransferx2(%24, %26) : (memref<?xf32>,  memref<?xf32>) -> ()
  call @printMemrefF32(%cast1) : (memref<*xf32>) -> ()

  // CHECK: [2.46, 2.46, 2.46, 2.46]
  call @vectransferx4(%24, %26) : (memref<?xf32>,  memref<?xf32>) -> ()
  call @printMemrefF32(%cast1) : (memref<*xf32>) -> ()
  return
}

func.func private @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func.func private @printMemrefF32(%ptr : memref<*xf32>)
