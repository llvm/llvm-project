// RUN: mlir-opt --test-gpu-rewrite -split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @shuffleF64
  // CHECK-SAME: (%[[SZ:.*]]: index, %[[VALUE:.*]]: f64, %[[OFF:.*]]: i32, %[[WIDTH:.*]]: i32, %[[MEM:.*]]: memref<f64, 1>) {
  func.func @shuffleF64(%sz : index, %value: f64, %offset: i32, %width: i32, %mem: memref<f64, 1>) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
               threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
      // CHECK: %[[INTVAL:.*]] = arith.bitcast %[[VALUE]] : f64 to i64
      // CHECK-NEXT: %[[LO:.*]] = arith.trunci %[[INTVAL]] : i64 to i32
      // CHECK-NEXT: %[[HI64:.*]] = arith.shrui %[[INTVAL]], %[[C32:.*]] : i64
      // CHECK-NEXT: %[[HI:.*]] = arith.trunci %[[HI64]] : i64 to i32
      // CHECK-NEXT: %[[SH1:.*]], %[[V1:.*]] = gpu.shuffle  xor %[[LO]], %[[OFF]], %[[WIDTH]] : i32
      // CHECK-NEXT: %[[SH2:.*]], %[[V2:.*]] = gpu.shuffle  xor %[[HI]], %[[OFF]], %[[WIDTH]] : i32
      // CHECK-NEXT: %[[LOSH:.*]] = arith.extui %[[SH1]] : i32 to i64
      // CHECK-NEXT: %[[HISHTMP:.*]] = arith.extui %[[SH2]] : i32 to i64
      // CHECK-NEXT: %[[HISH:.*]] = arith.shli %[[HISHTMP]], %[[C32]] : i64
      // CHECK-NEXT: %[[SHFLINT:.*]] = arith.ori %[[HISH]], %[[LOSH]] : i64
      // CHECK-NEXT:  = arith.bitcast %[[SHFLINT]] : i64 to f64
      %shfl, %pred = gpu.shuffle xor %value, %offset, %width : f64
      memref.store %shfl, %mem[]  : memref<f64, 1>
      gpu.terminator
    }
    return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @shuffleI64
  // CHECK-SAME: (%[[SZ:.*]]: index, %[[VALUE:.*]]: i64, %[[OFF:.*]]: i32, %[[WIDTH:.*]]: i32, %[[MEM:.*]]: memref<i64, 1>) {
  func.func @shuffleI64(%sz : index, %value: i64, %offset: i32, %width: i32, %mem: memref<i64, 1>) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
               threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
      // CHECK: %[[LO:.*]] = arith.trunci %[[VALUE]] : i64 to i32
      // CHECK-NEXT: %[[HI64:.*]] = arith.shrui %[[VALUE]], %[[C32:.*]] : i64
      // CHECK-NEXT: %[[HI:.*]] = arith.trunci %[[HI64]] : i64 to i32
      // CHECK-NEXT: %[[SH1:.*]], %[[V1:.*]] = gpu.shuffle  xor %[[LO]], %[[OFF]], %[[WIDTH]] : i32
      // CHECK-NEXT: %[[SH2:.*]], %[[V2:.*]] = gpu.shuffle  xor %[[HI]], %[[OFF]], %[[WIDTH]] : i32
      // CHECK-NEXT: %[[LOSH:.*]] = arith.extui %[[SH1]] : i32 to i64
      // CHECK-NEXT: %[[HISHTMP:.*]] = arith.extui %[[SH2]] : i32 to i64
      // CHECK-NEXT: %[[HISH:.*]] = arith.shli %[[HISHTMP]], %[[C32]] : i64
      // CHECK-NEXT: %[[SHFLINT:.*]] = arith.ori %[[HISH]], %[[LOSH]] : i64
      %shfl, %pred = gpu.shuffle xor %value, %offset, %width : i64
      memref.store %shfl, %mem[]  : memref<i64, 1>
      gpu.terminator
    }
    return
  }
}
