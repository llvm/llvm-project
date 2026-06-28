// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck %s --check-prefixes=ALL,DEFAULT
// RUN: mlir-opt %s -convert-vector-to-llvm='enable-gep-inbounds-nuw=1' -split-input-file | FileCheck %s --check-prefixes=ALL,INBOUNDS

// Verify that ConvertVectorToLLVMPass respects the module's data layout when
// deriving the index type. When the module declares a 32-bit index (via
// dlti.dl_spec), GEP arithmetic should be emitted in i32 instead of the
// default i64.  This also makes enable-gep-inbounds-nuw meaningful on 32-bit
// targets: the flag emits nsw/nuw on the *narrow* i32 multiply, allowing SCEV
// to form a clean 32-bit AddRec and enabling LSR to use direct pointer
// induction instead of per-iteration sign/zero extension.

// -----

// 32-bit data layout: index -> i32 in GEP arithmetic.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @load_32bit_index(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

}

// ALL-LABEL: func @load_32bit_index
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i32
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// ALL: llvm.load %[[GEP]] {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>

// -----

// With enable-gep-inbounds-nuw, the narrow i32 multiply and add carry
// nsw/nuw flags, enabling SCEV to form a clean 32-bit AddRec.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @load_32bit_index_nuw_mul(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

}

// DEFAULT-LABEL: func @load_32bit_index_nuw_mul
// DEFAULT: llvm.mul %{{.*}}, %{{.*}} : i32
// DEFAULT: llvm.add %{{.*}}, %{{.*}} : i32
// DEFAULT: llvm.getelementptr %{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, f32

// INBOUNDS-LABEL: func @load_32bit_index_nuw_mul
// INBOUNDS: llvm.mul %{{.*}}, %{{.*}} overflow<nsw, nuw> : i32
// INBOUNDS: llvm.add %{{.*}}, %{{.*}} overflow<nsw, nuw> : i32
// INBOUNDS: llvm.getelementptr inbounds|nuw %{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, f32

// -----

// Same test for store.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @store_32bit_index(%memref : memref<200x100xf32>, %i : index, %j : index, %val : vector<8xf32>) {
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return
}

}

// ALL-LABEL: func @store_32bit_index
// ALL: %[[C100:.*]] = llvm.mlir.constant(100 : index) : i32
// ALL: %[[MUL:.*]] = llvm.mul %{{.*}}, %[[C100]]
// ALL: %[[ADD:.*]] = llvm.add %[[MUL]], %{{.*}}
// DEFAULT: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ADD]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// INBOUNDS: %[[GEP:.*]] = llvm.getelementptr inbounds|nuw %{{.*}}[%[[ADD]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// ALL: llvm.store %{{.*}}, %[[GEP]] {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
