// RUN: mlir-opt --canonicalize -split-input-file %s | FileCheck %s

llvm.mlir.global internal unnamed_addr constant @foo(dense<[1.5903078570611027E-10, -2.5050911383645487E-8, 2.7557314984630029E-6, -1.984126983447703E-4, 0.0083333333333293485, -0.16666666666666663, 0.000000e+00, 0.000000e+00, -1.1367817304626284E-11, 2.08758833785978E-9, -2.7557315542999557E-7, 2.4801587293618683E-5, -0.0013888888888880667, 0.041666666666666637, -5.000000e-01, 1.000000e+00]> : tensor<16xf64>) {addr_space = 1 : i32, alignment = 8 : i64, dso_local} : !llvm.array<16 x f64>

// CHECK-LABEL: func @constant_load_idx(
// CHECK-SAME:    %[[IDX:.*]]: i64
func.func @constant_load_idx(%idx: i64) -> f64 {
  // CHECK: %[[C9:.*]] = arith.constant 9
  %c8 = arith.constant 8 : index

  // CHECK: %[[BASE:.*]] = llvm.mlir.addressof
  %ptr = llvm.mlir.addressof @foo : !llvm.ptr<1>

  // CHECK-NOT: llvm.getelementptr
  %ptr_i8 = llvm.getelementptr inbounds %ptr[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
  %ptr_offset = llvm.getelementptr inbounds %ptr_i8[8] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"(%[[BASE]])
  %memref = "polygeist.pointer2memref"(%ptr_offset) : (!llvm.ptr<1>) -> memref<?xf64>

  // CHECK: %[[IDX_CAST:.*]] = arith.index_cast %[[IDX]]
  // CHECK: %[[OFFSET:.*]] = arith.addi %[[IDX_CAST]], %[[C9]] : index
  // CHECK: memref.load %[[MEMREF]][%[[OFFSET]]] : memref<?xf64>
  %val = memref.load %memref[%c8] : memref<?xf64>

  func.return %val : f64
}

// -----

// CHECK-LABEL: func @dynamic_load_idx(
// CHECK-SAME:    %[[IDX1:.*]]: i64
// CHECK-SAME:    %[[IDX2:.*]]: i32
// CHECK-SAME:    %[[IDX3:.*]]: i8
// CHECK-SAME:    %[[LOAD_IDX:.*]]: index
func.func @dynamic_load_idx(%idx1: i64, %idx2: i32, %idx3: i8, %load_idx: index) -> f16 {
  // CHECK: %[[C2:.*]] = arith.constant 2
  %c16 = llvm.mlir.constant(16: i32) : i32

  // CHECK: %[[BASE:.*]] = llvm.alloca
  %ptr = llvm.alloca %c16 x f16 {alignment = 4 : i64} : (i32) -> !llvm.ptr

  // CHECK-NOT: llvm.getelementptr
  %ptr_f32 = llvm.getelementptr %ptr[%idx1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %ptr_i16 = llvm.getelementptr %ptr_f32[%idx2] : (!llvm.ptr, i32) -> !llvm.ptr, i16
  %ptr_i8 = llvm.getelementptr inbounds %ptr_i16[%idx3] : (!llvm.ptr, i8) -> !llvm.ptr, i8

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"(%[[BASE]])
  %memref = "polygeist.pointer2memref"(%ptr_i8) : (!llvm.ptr) -> memref<?xf16>

  // CHECK: %[[IDX_CAST1:.*]] = arith.index_cast %[[IDX1]]
  // CHECK: %[[SCALED1:.*]] = arith.muli %[[IDX_CAST1]], %[[C2]] : index

  // CHECK: %[[IDX_CAST2:.*]] = arith.index_cast %[[IDX2]]
  // CHECK: %[[OFFSET2:.*]] = arith.addi %[[SCALED1]], %[[IDX_CAST2]] : index

  // CHECK: %[[IDX_CAST3:.*]] = arith.index_cast %[[IDX3]]
  // CHECK: %[[SCALED3:.*]] = arith.divsi %[[IDX_CAST3]], %[[C2]] : index
  // CHECK: %[[OFFSET3:.*]] = arith.addi %[[OFFSET2]], %[[SCALED3]] : index
  
  // CHECK: %[[EIDX:.*]] = arith.addi %[[LOAD_IDX]], %[[OFFSET3]] : index

  // CHECK: memref.load %[[MEMREF]][%[[EIDX]]] : memref<?xf16>
  %val = memref.load %memref[%load_idx] : memref<?xf16>

  func.return %val : f16
}

// -----

// CHECK-LABEL: func @reject_unaligned_gep_cst_idx(
// CHECK-SAME:    %[[LOAD_IDX:.*]]: index
func.func @reject_unaligned_gep_cst_idx(%load_idx: index) -> f16 {
  %c1 = arith.constant 1 : i32
  %c16 = llvm.mlir.constant(16: i32) : i32

  %ptr = llvm.alloca %c16 x f16 {alignment = 4 : i64} : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr
  %ptr_i8 = llvm.getelementptr %ptr[%c1] : (!llvm.ptr, i32) -> !llvm.ptr, i8

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"(%[[GEP]])
  %memref = "polygeist.pointer2memref"(%ptr_i8) : (!llvm.ptr) -> memref<?xf16>

  // CHECK: memref.load %[[MEMREF]][%[[LOAD_IDX]]] : memref<?xf16>
  %val = memref.load %memref[%load_idx] : memref<?xf16>

  func.return %val : f16
}

// -----

// CHECK-LABEL: func @reject_unaligned_gep_cst_scalar(
// CHECK-SAME:    %[[LOAD_IDX:.*]]: index
func.func @reject_unaligned_gep_cst_scalar(%load_idx: index) -> f16 {
  %c16 = llvm.mlir.constant(16: i32) : i32

  %ptr = llvm.alloca %c16 x f16 {alignment = 4 : i64} : (i32) -> !llvm.ptr

  // CHECK: %[[GEP:.*]] = llvm.getelementptr
  %ptr_i8 = llvm.getelementptr %ptr[1] : (!llvm.ptr) -> !llvm.ptr, i8

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"(%[[GEP]])
  %memref = "polygeist.pointer2memref"(%ptr_i8) : (!llvm.ptr) -> memref<?xf16>

  // CHECK: memref.load %[[MEMREF]][%[[LOAD_IDX]]] : memref<?xf16>
  %val = memref.load %memref[%load_idx] : memref<?xf16>

  func.return %val : f16
}

// -----

// CHECK-LABEL: func @array_load_idx(
// CHECK-SAME:    %[[IDX:.*]]: i64
func.func @array_load_idx(%idx: i64) -> f64 {
  %c0 = arith.constant 0 : index
  %c16 = llvm.mlir.constant(16: i32) : i32

  %ptr = llvm.alloca %c16 x !llvm.array<8 x i8> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  // CHECK-NOT: llvm.getelementptr
  %ptr_array = llvm.getelementptr inbounds %ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"
  %memref = "polygeist.pointer2memref"(%ptr_array) : (!llvm.ptr) -> memref<?xf64>

  // CHECK: %[[IDX_CAST:.*]] = arith.index_cast %[[IDX]]
  // CHECK: memref.load %[[MEMREF]][%[[IDX_CAST]]] : memref<?xf64>
  %val = memref.load %memref[%c0] : memref<?xf64>

  func.return %val : f64
}

// -----

llvm.mlir.global internal unnamed_addr constant @foo(dense<[1.5903078570611027E-10, -2.5050911383645487E-8, 2.7557314984630029E-6, -1.984126983447703E-4, 0.0083333333333293485, -0.16666666666666663, 0.000000e+00, 0.000000e+00, -1.1367817304626284E-11, 2.08758833785978E-9, -2.7557315542999557E-7, 2.4801587293618683E-5, -0.0013888888888880667, 0.041666666666666637, -5.000000e-01, 1.000000e+00]> : tensor<16xf64>) {addr_space = 1 : i32, alignment = 8 : i64, dso_local} : !llvm.array<16 x f64>

// CHECK-LABEL: func @constant_store_idx(
// CHECK-SAME:    %[[IDX:.*]]: i64, %[[VAL:.*]]: f64
func.func @constant_store_idx(%idx: i64, %val: f64) {
  // CHECK: %[[C9:.*]] = arith.constant 9
  %c8 = arith.constant 8 : index

  // CHECK: %[[BASE:.*]] = llvm.mlir.addressof
  %ptr = llvm.mlir.addressof @foo : !llvm.ptr<1>

  // CHECK-NOT: llvm.getelementptr
  %ptr_i8 = llvm.getelementptr inbounds %ptr[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
  %ptr_offset = llvm.getelementptr inbounds %ptr_i8[8] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"(%[[BASE]])
  %memref = "polygeist.pointer2memref"(%ptr_offset) : (!llvm.ptr<1>) -> memref<?xf64>

  // CHECK: %[[IDX_CAST:.*]] = arith.index_cast %[[IDX]]
  // CHECK: %[[OFFSET:.*]] = arith.addi %[[IDX_CAST]], %[[C9]] : index
  // CHECK: memref.store %[[VAL]], %[[MEMREF]][%[[OFFSET]]] : memref<?xf64>
  memref.store %val, %memref[%c8] : memref<?xf64>

  func.return
}

// -----

// CHECK-LABEL: func @dynamic_store_idx(
// CHECK-SAME:    %[[IDX1:.*]]: i64
// CHECK-SAME:    %[[IDX2:.*]]: i32
// CHECK-SAME:    %[[IDX3:.*]]: i8
// CHECK-SAME:    %[[LOAD_IDX:.*]]: index
// CHECK-SAME:    %[[VAL:.*]]: f16
func.func @dynamic_store_idx(%idx1: i64, %idx2: i32, %idx3: i8, %load_idx: index, %val: f16) {
  // CHECK: %[[C2:.*]] = arith.constant 2
  %c16 = llvm.mlir.constant(16: i32) : i32

  // CHECK: %[[BASE:.*]] = llvm.alloca
  %ptr = llvm.alloca %c16 x f16 {alignment = 4 : i64} : (i32) -> !llvm.ptr

  // CHECK-NOT: llvm.getelementptr
  %ptr_f32 = llvm.getelementptr %ptr[%idx1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %ptr_i16 = llvm.getelementptr %ptr_f32[%idx2] : (!llvm.ptr, i32) -> !llvm.ptr, i16
  %ptr_i8 = llvm.getelementptr inbounds %ptr_i16[%idx3] : (!llvm.ptr, i8) -> !llvm.ptr, i8

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"(%[[BASE]])
  %memref = "polygeist.pointer2memref"(%ptr_i8) : (!llvm.ptr) -> memref<?xf16>

  // CHECK: %[[IDX_CAST1:.*]] = arith.index_cast %[[IDX1]]
  // CHECK: %[[SCALED1:.*]] = arith.muli %[[IDX_CAST1]], %[[C2]] : index

  // CHECK: %[[IDX_CAST2:.*]] = arith.index_cast %[[IDX2]]
  // CHECK: %[[OFFSET2:.*]] = arith.addi %[[SCALED1]], %[[IDX_CAST2]] : index

  // CHECK: %[[IDX_CAST3:.*]] = arith.index_cast %[[IDX3]]
  // CHECK: %[[SCALED3:.*]] = arith.divsi %[[IDX_CAST3]], %[[C2]] : index
  // CHECK: %[[OFFSET3:.*]] = arith.addi %[[OFFSET2]], %[[SCALED3]] : index
  
  // CHECK: %[[EIDX:.*]] = arith.addi %[[LOAD_IDX]], %[[OFFSET3]] : index

  // CHECK: memref.store %[[VAL]], %[[MEMREF]][%[[EIDX]]] : memref<?xf16>
  memref.store %val, %memref[%load_idx] : memref<?xf16>

  func.return
}

// -----

// CHECK-LABEL: func @array_store_idx(
// CHECK-SAME:    %[[IDX:.*]]: i64
// CHECK-SAME:    %[[VAL:.*]]: f64
func.func @array_store_idx(%idx: i64, %val: f64) {
  %c0 = arith.constant 0 : index
  %c16 = llvm.mlir.constant(16: i32) : i32

  %ptr = llvm.alloca %c16 x !llvm.array<8 x i8> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  // CHECK-NOT: llvm.getelementptr
  %ptr_array = llvm.getelementptr inbounds %ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>

  // CHECK: %[[MEMREF:.*]] = "polygeist.pointer2memref"
  %memref = "polygeist.pointer2memref"(%ptr_array) : (!llvm.ptr) -> memref<?xf64>

  // CHECK: %[[IDX_CAST:.*]] = arith.index_cast %[[IDX]]
  // CHECK: memref.store %[[VAL]], %[[MEMREF]][%[[IDX_CAST]]] : memref<?xf64>
  memref.store %val, %memref[%c0] : memref<?xf64>

  func.return
}
