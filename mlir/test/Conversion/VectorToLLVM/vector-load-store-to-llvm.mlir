// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s --check-prefixes=ALL,DEFAULT
// RUN: mlir-opt %s -convert-vector-to-llvm='enable-gep-inbounds-nuw=1' -split-input-file | FileCheck %s --check-prefixes=ALL,INBOUNDS

//===----------------------------------------------------------------------===//
// vector.load
//===----------------------------------------------------------------------===//

func.func @load(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// ALL-LABEL: func @load
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.load %[[GEP]] {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>

// -----

func.func @load_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<[8]xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<[8]xf32>
  return %0 : vector<[8]xf32>
}

// ALL-LABEL: func @load_scalable
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.load %[[GEP]] {alignment = 4 : i64} : !llvm.ptr -> vector<[8]xf32>

// -----

func.func @load_nontemporal(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// ALL-LABEL: func @load_nontemporal
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.load %[[GEP]] {alignment = 4 : i64, nontemporal} : !llvm.ptr -> vector<8xf32>

// -----

func.func @load_nontemporal_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<[8]xf32> {
  %0 = vector.load %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<[8]xf32>
  return %0 : vector<[8]xf32>
}

// ALL-LABEL: func @load_nontemporal_scalable
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.load %[[GEP]] {alignment = 4 : i64, nontemporal} : !llvm.ptr -> vector<[8]xf32>

// -----

func.func @load_index(%memref : memref<200x100xindex>, %i : index, %j : index) -> vector<8xindex> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xindex>, vector<8xindex>
  return %0 : vector<8xindex>
}
// ALL-LABEL: func @load_index
// ALL: %[[T0:.*]] = llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr -> vector<8xi64>
// ALL: %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<8xi64> to vector<8xindex>
// ALL: return %[[T1]] : vector<8xindex>

// -----

func.func @load_index_scalable(%memref : memref<200x100xindex>, %i : index, %j : index) -> vector<[8]xindex> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xindex>, vector<[8]xindex>
  return %0 : vector<[8]xindex>
}
// ALL-LABEL: func @load_index_scalable
// ALL: %[[T0:.*]] = llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr -> vector<[8]xi64>
// ALL: %[[T1:.*]] = builtin.unrealized_conversion_cast %[[T0]] : vector<[8]xi64> to vector<[8]xindex>
// ALL: return %[[T1]] : vector<[8]xindex>

// -----

func.func @load_0d(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<f32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<f32>
  return %0 : vector<f32>
}

// ALL-LABEL: func @load_0d
// ALL: %[[J:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// ALL: %[[I:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// ALL: %[[CAST_MEMREF:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<200x100xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// ALL: %[[REF:.*]] = llvm.extractvalue %[[CAST_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %[[I]], %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %[[J]]
// DEFAULT: %[[ADDR:.*]] = llvm.getelementptr %[[REF]][%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[ADDR:.*]] = llvm.getelementptr inbounds|nuw %[[REF]][%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: %[[LOAD:.*]] = llvm.load %[[ADDR]] {alignment = 4 : i64} : !llvm.ptr -> vector<1xf32>
// ALL: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[LOAD]] : vector<1xf32> to vector<f32>
// ALL: return %[[RES]] : vector<f32>

// -----

func.func @load_with_alignment(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] { alignment = 8 } : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// ALL-LABEL: func @load_with_alignment
// ALL: llvm.load {{.*}} {alignment = 8 : i64} : !llvm.ptr -> vector<8xf32>

// -----

//===----------------------------------------------------------------------===//
// vector.store
//===----------------------------------------------------------------------===//

func.func @store(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<4xf32>
  return
}

// ALL-LABEL: func @store
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64} :  vector<4xf32>, !llvm.ptr

// -----

func.func @store_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<[4]xf32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<[4]xf32>
  return
}

// ALL-LABEL: func @store_scalable
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64} :  vector<[4]xf32>, !llvm.ptr

// -----

func.func @store_nontemporal(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<4xf32>
  return
}

// ALL-LABEL: func @store_nontemporal
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64, nontemporal} :  vector<4xf32>, !llvm.ptr

// -----

func.func @store_nontemporal_scalable(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<[4]xf32>
  vector.store %val, %memref[%i, %j] {nontemporal = true} : memref<200x100xf32>, vector<[4]xf32>
  return
}

// ALL-LABEL: func @store_nontemporal_scalable
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64, nontemporal} :  vector<[4]xf32>, !llvm.ptr

// -----

func.func @store_index(%memref : memref<200x100xindex>, %i : index, %j : index) {
  %val = arith.constant dense<11> : vector<4xindex>
  vector.store %val, %memref[%i, %j] : memref<200x100xindex>, vector<4xindex>
  return
}
// ALL-LABEL: func @store_index
// ALL: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<4xi64>, !llvm.ptr

// -----

func.func @store_index_scalable(%memref : memref<200x100xindex>, %i : index, %j : index) {
  %val = arith.constant dense<11> : vector<[4]xindex>
  vector.store %val, %memref[%i, %j] : memref<200x100xindex>, vector<[4]xindex>
  return
}
// ALL-LABEL: func @store_index_scalable
// ALL: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<[4]xi64>, !llvm.ptr

// -----

func.func @store_0d(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<f32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<f32>
  return
}

// ALL-LABEL: func @store_0d
// ALL: %[[J:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// ALL: %[[I:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
// ALL: %[[CAST_MEMREF:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<200x100xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// ALL: %[[CST:.*]] = arith.constant dense<1.100000e+01> : vector<f32>
// ALL: %[[VAL:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<f32> to vector<1xf32>
// ALL: %[[REF:.*]] = llvm.extractvalue %[[CAST_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i64
// ALL: %[[MUL:.*]] = llvm.mul %[[I]], %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %[[J]]
// DEFAULT: %[[ADDR:.*]] = llvm.getelementptr %[[REF]][%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// INBOUNDS: %[[ADDR:.*]] = llvm.getelementptr inbounds|nuw %[[REF]][%[[ADD]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALL: llvm.store %[[VAL]], %[[ADDR]] {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr
// ALL: return

// -----

func.func @store_with_alignment(%memref : memref<200x100xf32>, %i : index, %j : index, %val : vector<4xf32>) {
  vector.store %val, %memref[%i, %j] {alignment = 8} : memref<200x100xf32>, vector<4xf32>
  return
}

// ALL-LABEL: func @store_with_alignment
// ALL: llvm.store %{{.*}} {alignment = 8 : i64} :  vector<4xf32>, !llvm.ptr
