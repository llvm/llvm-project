// RUN: mlir-opt %s --convert-vector-to-llvm='use-vector-alignment=0' --split-input-file | FileCheck %s --check-prefix=MEMREF-ALIGN
// RUN: mlir-opt %s --convert-vector-to-llvm='use-vector-alignment=1' --split-input-file | FileCheck %s --check-prefix=VEC-ALIGN


//===----------------------------------------------------------------------===//
// vector.load
//===----------------------------------------------------------------------===//

func.func @load(%base : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %base[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// ALL-LABEL: func @load

// VEC-ALIGN: llvm.load %{{.*}} {alignment = 32 : i64} : !llvm.ptr -> vector<8xf32>
// MEMREF-ALIGN: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>

// -----

func.func @load_with_alignment_attribute(%base : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %base[%i, %j] {alignment = 8} : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// ALL-LABEL: func @load_with_alignment_attribute

// VEC-ALIGN: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr -> vector<8xf32>
// MEMREF-ALIGN: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr -> vector<8xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.store
//===----------------------------------------------------------------------===//

func.func @store(%base : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %base[%i, %j] : memref<200x100xf32>, vector<4xf32>
  return
}

// ALL-LABEL: func @store

// VEC-ALIGN: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} :  vector<4xf32>, !llvm.ptr
// MEMREF-ALIGN: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} :  vector<4xf32>, !llvm.ptr

// -----

func.func @store_with_alignment_attribute(%base : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %base[%i, %j] {alignment = 8} : memref<200x100xf32>, vector<4xf32>
  return
}

// ALL-LABEL: func @store_with_alignment_attribute

// VEC-ALIGN: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} :  vector<4xf32>, !llvm.ptr
// MEMREF-ALIGN: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} :  vector<4xf32>, !llvm.ptr

// -----

//===----------------------------------------------------------------------===//
// vector.maskedload
//===----------------------------------------------------------------------===//

func.func @masked_load(%base: memref<?xf32>, %mask: vector<16xi1>, %passthru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %base[%c0], %mask, %passthru : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %0 : vector<16xf32>
}

// ALL-LABEL: func @masked_load

// VEC-ALIGN: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 64 : i32} : (!llvm.ptr, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
// MEMREF-ALIGN: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.ptr, vector<16xi1>, vector<16xf32>) -> vector<16xf32>

// -----

func.func @masked_load_with_alignment_attribute(%base: memref<?xf32>, %mask: vector<16xi1>, %passthru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %base[%c0], %mask, %passthru {alignment = 8} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %0 : vector<16xf32>
}

// ALL-LABEL: func @masked_load_with_alignment_attribute

// VEC-ALIGN: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.ptr, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
// MEMREF-ALIGN: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (!llvm.ptr, vector<16xi1>, vector<16xf32>) -> vector<16xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.maskedstore
//===----------------------------------------------------------------------===//

func.func @masked_store(%base: memref<?xf32>, %mask: vector<16xi1>, %passthru: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %base[%c0], %mask, %passthru : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// ALL-LABEL: func @masked_store

// VEC-ALIGN: llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 64 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr
// MEMREF-ALIGN: llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr

// -----

func.func @masked_store_with_alignment_attribute(%base: memref<?xf32>, %mask: vector<16xi1>, %passthru: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %base[%c0], %mask, %passthru {alignment = 8} : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// ALL-LABEL: func @masked_store_with_alignment_attribute

// VEC-ALIGN: llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr
// MEMREF-ALIGN: llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr

// -----

//===----------------------------------------------------------------------===//
// vector.scatter
//===----------------------------------------------------------------------===//

func.func @scatter(%base: memref<?xf32>, %index: vector<3xi32>, %mask: vector<3xi1>, %value: vector<3xf32>) {
  %0 = arith.constant 0: index
  vector.scatter %base[%0][%index], %mask, %value : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32>
  return
}

// ALL-LABEL: func @scatter

// VEC-ALIGN: llvm.intr.masked.scatter %{{.*}}, %{{.*}}, %{{.*}} {alignment = 16 : i32} : vector<3xf32>, vector<3xi1> into vector<3x!llvm.ptr>
// MEMREF-ALIGN: llvm.intr.masked.scatter %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : vector<3xf32>, vector<3xi1> into vector<3x!llvm.ptr>

// -----

func.func @scatter_with_alignment_attribute(%base: memref<?xf32>, %index: vector<3xi32>, %mask: vector<3xi1>, %value: vector<3xf32>) {
  %0 = arith.constant 0: index
  vector.scatter %base[%0][%index], %mask, %value {alignment = 8} : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32>
  return
}

// ALL-LABEL: func @scatter_with_alignment_attribute

// VEC-ALIGN: llvm.intr.masked.scatter %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : vector<3xf32>, vector<3xi1> into vector<3x!llvm.ptr>
// MEMREF-ALIGN: llvm.intr.masked.scatter %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : vector<3xf32>, vector<3xi1> into vector<3x!llvm.ptr>

// -----

//===----------------------------------------------------------------------===//
// vector.gather
//===----------------------------------------------------------------------===//

func.func @gather(%base: memref<?xf32>, %index: vector<3xi32>, %mask: vector<3xi1>, %passthru: vector<3xf32>) -> vector<3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %base[%0][%index], %mask, %passthru : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// ALL-LABEL: func @gather

// VEC-ALIGN: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 16 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// MEMREF-ALIGN: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 4 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>

// -----

func.func @gather_with_alignment_attribute(%base: memref<?xf32>, %index: vector<3xi32>, %mask: vector<3xi1>, %passthru: vector<3xf32>) -> vector<3xf32> {
  %0 = arith.constant 0: index
  %1 = vector.gather %base[%0][%index], %mask, %passthru {alignment = 8} : memref<?xf32>, vector<3xi32>, vector<3xi1>, vector<3xf32> into vector<3xf32>
  return %1 : vector<3xf32>
}

// ALL-LABEL: func @gather_with_alignment_attribute

// VEC-ALIGN: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
// MEMREF-ALIGN: %[[G:.*]] = llvm.intr.masked.gather %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} : (vector<3x!llvm.ptr>, vector<3xi1>, vector<3xf32>) -> vector<3xf32>
