// RUN: mlir-opt %s -split-input-file --pass-pipeline='builtin.module(func.func(nvgpu-optimize-shared-memory))' | FileCheck %s

// CHECK: @optimize_128x32xf16_32x128xf16([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @optimize_128x32xf16_32x128xf16(%arg0: memref<128x128xf16>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> (vector<4x2xf16>, vector<4x2xf16>) {
  // CHECK: [[shm:%.+]] = memref.alloc
  // CHECK: [[shmB:%.+]] = memref.alloc
  %shm = memref.alloc() : memref<128x32xf16, 3>
  %shmB = memref.alloc() : memref<32x128xf16, 3>

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c6]]
  // CHECK: [[c2:%.+]] = arith.constant 2 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[src_bits]], [[c2]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stColPerm]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 8
      : memref<128x128xf16> to memref<128x32xf16, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c2:%.+]] = arith.constant 2 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: nvgpu.ldmatrix [[shm]][[[fragRow]], [[fragColPerm]]]
  %mat = nvgpu.ldmatrix %shm[%fragRow, %fragCol] {numTiles = 4 : i32, transpose = false}
      : memref<128x32xf16, 3> -> vector<4x2xf16>

  // CHECK: [[c15:%.+]] = arith.constant 15 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c15]]
  // CHECK: [[c3:%.+]] = arith.constant 3 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[src_bits]], [[c3]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shmB]][[[stRow]], [[stColPerm]]]
  %2 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shmB[%stRow, %stCol], 8
      : memref<128x128xf16> to memref<32x128xf16, 3>
  %3 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c15:%.+]] = arith.constant 15 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c15]]
  // CHECK: [[c3:%.+]] = arith.constant 3 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c3]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: nvgpu.ldmatrix [[shmB]][[[fragRow]], [[fragColPerm]]]
  %matB = nvgpu.ldmatrix %shmB[%fragRow, %fragCol] {numTiles = 4 : i32, transpose = false}
      : memref<32x128xf16, 3> -> vector<4x2xf16>

  return %mat, %matB: vector<4x2xf16>, vector<4x2xf16>
}


// -----

// CHECK: @optimize_64x16xf32_16x64xf32([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @optimize_64x16xf32_16x64xf32(%arg0: memref<128x128xf32>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> (vector<4x1xf32>, vector<4x1xf32>, f32, vector<4xf32>, f32) {
  // CHECK: [[shm:%.+]] = memref.alloc
  // CHECK: [[shmB:%.+]] = memref.alloc
  %shm = memref.alloc() : memref<64x16xf32, 3>
  %shmB = memref.alloc() : memref<16x64xf32, 3>

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[src_bits]], [[c1]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stColPerm]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 4
      : memref<128x128xf32> to memref<64x16xf32, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: nvgpu.ldmatrix [[shm]][[[fragRow]], [[fragColPerm]]]
  %mat = nvgpu.ldmatrix %shm[%fragRow, %fragCol] {numTiles = 4 : i32, transpose = false}
      : memref<64x16xf32, 3> -> vector<4x1xf32>

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: memref.load [[shm]][[[fragRow]], [[fragColPerm]]]
  %elem = memref.load %shm[%fragRow, %fragCol] : memref<64x16xf32, 3>

  // Verify vector operations.

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: vector.load [[shm]][[[fragRow]], [[fragColPerm]]]
  %elem2 = vector.load %shm[%fragRow, %fragCol] : memref<64x16xf32, 3>, vector<4xf32>

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: vector.store %{{.+}}, [[shm]][[[fragRow]], [[fragColPerm]]]
  vector.store %elem2, %shm[%fragRow, %fragCol] : memref<64x16xf32, 3>, vector<4xf32>

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: memref.store %{{.+}}, [[shm]][[[fragRow]], [[fragColPerm]]]
  memref.store %elem, %shm[%fragRow, %fragCol] : memref<64x16xf32, 3>

  // Verify 16x64xf32 memory size.

  // CHECK: [[c15:%.+]] = arith.constant 15 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c15]]
  // CHECK: [[c2:%.+]] = arith.constant 2 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[src_bits]], [[c2]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shmB]][[[stRow]], [[stColPerm]]]
  %2 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shmB[%stRow, %stCol], 4
      : memref<128x128xf32> to memref<16x64xf32, 3>
  %3 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c15:%.+]] = arith.constant 15 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c15]]
  // CHECK: [[c2:%.+]] = arith.constant 2 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: nvgpu.ldmatrix [[shmB]][[[fragRow]], [[fragColPerm]]]
  %matB = nvgpu.ldmatrix %shmB[%fragRow, %fragCol] {numTiles = 4 : i32, transpose = false}
      : memref<16x64xf32, 3> -> vector<4x1xf32>

  // CHECK: [[c15:%.+]] = arith.constant 15 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c15]]
  // CHECK: [[c2:%.+]] = arith.constant 2 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: memref.load [[shmB]][[[fragRow]], [[fragColPerm]]]
  %elemB = memref.load %shmB[%fragRow, %fragCol] : memref<16x64xf32, 3>

  return %mat, %matB, %elem, %elem2, %elemB: vector<4x1xf32>, vector<4x1xf32>, f32, vector<4xf32>, f32
}


// -----

// Small column edge cases

// CHECK: @small_column_size_f64([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @small_column_size_f64(%arg0: memref<32x32xf64>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> f64 {
  // CHECK: [[shm:%.+]] = memref.alloc
  %shm = memref.alloc() : memref<32x4xf64, 3>

  // CHECK: [[c4:%.+]] = arith.constant 4 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c4]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shrui [[src_bits]], [[c1]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stColPerm]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 2
      : memref<32x32xf64> to memref<32x4xf64, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c6:%.+]] = arith.constant 4 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shrui [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: memref.load [[shm]][[[fragRow]], [[fragColPerm]]]
  %el = memref.load %shm[%fragRow, %fragCol] : memref<32x4xf64, 3>

  return %el: f64
}

// CHECK: @too_small_column_size_f16([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @too_small_column_size_f16(%arg0: memref<128x128xf16>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> vector<1x2xf16> {
  // CHECK: [[shm:%.+]] = memref.alloc
  %shm = memref.alloc() : memref<128x8xf16, 3>

  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stCol]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 8
      : memref<128x128xf16> to memref<128x8xf16, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: nvgpu.ldmatrix [[shm]][[[fragRow]], [[fragCol]]]
  %mat = nvgpu.ldmatrix %shm[%fragRow, %fragCol] {numTiles = 1 : i32, transpose = false}
      : memref<128x8xf16, 3> -> vector<1x2xf16>

  return %mat: vector<1x2xf16>
}

// -----

// CHECK: @abort_if_subview([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @abort_if_subview(%arg0: memref<128x128xf16>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> vector<1x2xf16> {
  // CHECK: [[shm:%.+]] = memref.alloc
  %shm = memref.alloc() : memref<128x32xf16, 3>
  // CHECK: [[shmView:%.+]] = memref.subview
  %shmView = memref.subview %shm[0, 0][64, 32][1, 1] : memref<128x32xf16, 3> to memref<64x32xf16, 3>

  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stCol]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 8
      : memref<128x128xf16> to memref<128x32xf16, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: nvgpu.ldmatrix [[shmView]][[[fragRow]], [[fragCol]]]
  %mat = nvgpu.ldmatrix %shmView[%fragRow, %fragCol] {numTiles = 1 : i32, transpose = false}
      : memref<64x32xf16, 3> -> vector<1x2xf16>

  return %mat: vector<1x2xf16>
}
