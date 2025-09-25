// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

gpu.module @xevm_module {
gpu.func @store_1D_vector(%vec: vector<8xf32>, %source: memref<8x16x32xf32>,
     %off1: index, %off2: index, %off3: index,
     %indices: vector<8xindex>, %mask: vector<8xi1>) {
  vector.scatter %source[%off1, %off2, %off3][%indices], %mask, %vec
       : memref<8x16x32xf32>, vector<8xindex>, vector<8xi1>, vector<8xf32>
  gpu.return
}
// CHECK-LABEL:  @store_1D_vector(
// CHECK-SAME:   %[[VAL:.+]]: vector<8xf32>, %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>, %[[MASK:.+]]: vector<8xi1>) {
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        xegpu.store %[[VAL]], %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8xf32>, i64, vector<8xindex>, vector<8xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @store_2D_memref(%vec: vector<8xf32>, %source: memref<8x32xf32>,
     %off1: index, %off2: index,
     %indices: vector<8xindex>, %mask: vector<8xi1>) {
  vector.scatter %source[%off1, %off2][%indices], %mask, %vec
       : memref<8x32xf32>, vector<8xindex>, vector<8xi1>, vector<8xf32>
  gpu.return
}
// CHECK-LABEL:  @store_2D_memref(
// CHECK-SAME:   %[[VAL:.+]]: vector<8xf32>, %[[SRC:.+]]: memref<8x32xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>, %[[MASK:.+]]: vector<8xi1>) {
// CHECK-COUNT1: arith.muli {{.*}} : index
// CHECK-COUNT1: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x32xf32> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        xegpu.store %[[VAL]], %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8xf32>, i64, vector<8xindex>, vector<8xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @store_2D_vector(%vec: vector<8x16xf32>, %source: memref<8x16x32xf32>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8x16xindex>, %mask: vector<8x16xi1>) {
  vector.scatter %source[%off0, %off1, %off2][%indices], %mask, %vec
       : memref<8x16x32xf32>, vector<8x16xindex>, vector<8x16xi1>, vector<8x16xf32>
  gpu.return
}
// CHECK-LABEL:  @store_2D_vector(
// CHECK-SAME:   %[[VAL:.+]]: vector<8x16xf32>, %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8x16xindex>, %[[MASK:.+]]: vector<8x16xi1>) {
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8x16xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8x16xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        xegpu.store %[[VAL]], %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8x16xf32>, i64, vector<8x16xindex>, vector<8x16xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @store_dynamic_source(%vec: vector<8x16xf32>, %source: memref<?x?x?xf32>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8x16xindex>, %mask: vector<8x16xi1>) {
  vector.scatter %source[%off0, %off1, %off2][%indices], %mask, %vec
       : memref<?x?x?xf32>, vector<8x16xindex>, vector<8x16xi1>, vector<8x16xf32>
  gpu.return
}
// CHECK-LABEL:  @store_dynamic_source(
// CHECK-SAME:   %[[VAL:.+]]: vector<8x16xf32>, %[[SRC:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8x16xindex>, %[[MASK:.+]]: vector<8x16xi1>) {
// CHECK:        memref.extract_strided_metadata %[[SRC]]
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8x16xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8x16xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<?x?x?xf32> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        xegpu.store %[[VAL]], %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8x16xf32>, i64, vector<8x16xindex>, vector<8x16xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @store_dynamic_source2(%vec: vector<8x16xf32>, %source: memref<?x8x16xf32>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8x16xindex>, %mask: vector<8x16xi1>) {
  vector.scatter %source[%off0, %off1, %off2][%indices], %mask, %vec
       : memref<?x8x16xf32>, vector<8x16xindex>, vector<8x16xi1>, vector<8x16xf32>
  gpu.return
}
// CHECK-LABEL:  @store_dynamic_source2(
// CHECK-SAME:   %[[VAL:.+]]: vector<8x16xf32>, %[[SRC:.+]]: memref<?x8x16xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8x16xindex>, %[[MASK:.+]]: vector<8x16xi1>) {
// CHECK-NOT:    memref.extract_strided_metadata %[[SRC]]
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8x16xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8x16xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<?x8x16xf32> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        xegpu.store %[[VAL]], %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8x16xf32>, i64, vector<8x16xindex>, vector<8x16xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @non_unit_inner_stride_1D(
    %vec: vector<8xf32>, %source: memref<32xf32, strided<[?], offset: ?>>,
    %off: index, %indices: vector<8xindex>, %mask: vector<8xi1>) {
  vector.scatter %source[%off][%indices], %mask, %vec
    : memref<32xf32, strided<[?], offset: ?>>, vector<8xindex>, vector<8xi1>, vector<8xf32>
  gpu.return
}
// CHECK-LABEL:  @non_unit_inner_stride_1D(
// CHECK-SAME:   %[[VAL:.+]]: vector<8xf32>, %[[SRC:.+]]: memref<32xf32, strided<[?], offset: ?>>,
// CHECK-SAME:   %[[OFF1:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>, %[[MASK:.+]]: vector<8xi1>) {
// CHECK:        %[[BB:.+]], %[[M_OFF:.+]], %[[SZ:.+]], %[[STRIDE:.+]] = memref.extract_strided_metadata %[[SRC]]
// CHECK:        arith.muli %[[OFF1]], %[[STRIDE]] : index
// CHECK:        arith.addi {{.*}} : index
// CHECK:        %[[STRD_VEC:.+]] = vector.broadcast %[[STRIDE]] : index to vector<8xindex>
// CHECK:        %[[STRD_INDICES:.+]] = arith.muli %[[STRD_VEC:.+]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[STRD_INDICES]] : vector<8xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<32xf32, strided<[?], offset: ?>> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        xegpu.store %[[VAL]], %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8xf32>, i64, vector<8xindex>, vector<8xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @non_unit_inner_stride_3D(
    %vec: vector<8xf32>,
    %source: memref<4x8x32xf32, strided<[?, 128, 2], offset: ?>>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8xindex>, %mask: vector<8xi1>) {
  vector.scatter %source[%off0, %off1, %off2][%indices], %mask, %vec
    : memref<4x8x32xf32, strided<[?, 128, 2], offset: ?>>,
      vector<8xindex>, vector<8xi1>, vector<8xf32>
  gpu.return
}
// CHECK-LABEL:  @non_unit_inner_stride_3D(
// CHECK-SAME:   %[[VAL:.+]]: vector<8xf32>, %[[SRC:.+]]: memref<4x8x32xf32, strided<[?, 128, 2], offset: ?>>,
// CHECK-SAME:   %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>, %[[MASK:.+]]: vector<8xi1>) {
// CHECK:        %[[BB:.+]], %[[M_OFF:.+]], %[[SIZES:.+]]:3, %[[STRIDES:.+]]:3 = memref.extract_strided_metadata %[[SRC]]
// CHECK:        arith.muli %[[OFF0]], %[[STRIDES]]#0 : index
// CHECK:        arith.addi {{.*}} : index
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[STRD_INDICES:.+]] = arith.muli {{.*}}%[[INDICES]]{{.*}} : vector<8xindex>
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}} : index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[STRD_INDICES]] : vector<8xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<4x8x32xf32, strided<[?, 128, 2], offset: ?>> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        xegpu.store %[[VAL]], %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8xf32>, i64, vector<8xindex>, vector<8xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @scatter_into_subview(%vals: vector<8xf16>,
                               %source: memref<4096x4096xf16>,
                               %memref_off: index, %off1: index, %off2: index,
                               %indices: vector<8xindex>,
                               %mask: vector<8xi1>) {
  %subview = memref.subview %source[%memref_off, %memref_off] [256, 256] [1, 1]
      : memref<4096x4096xf16>
        to memref<256x256xf16, strided<[4096, 1], offset: ?>>
  vector.scatter %subview[%off1, %off2][%indices], %mask, %vals
      : memref<256x256xf16, strided<[4096, 1], offset: ?>>,
        vector<8xindex>, vector<8xi1>, vector<8xf16>
  gpu.return
}
// CHECK-LABEL:  @scatter_into_subview(
// CHECK-SAME:   %[[VALS:.+]]: vector<8xf16>,
// CHECK-SAME:   %[[SRC:.+]]: memref<4096x4096xf16>,
// CHECK-SAME:   %[[MEMREF_OFF:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>, %[[MASK:.+]]: vector<8xi1>) {
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]][%[[MEMREF_OFF]], %[[MEMREF_OFF]]] [256, 256] [1, 1]
// CHECK:        %[[BB:.+]], %[[OFFSET:.+]],{{.*}},{{.*}} = memref.extract_strided_metadata %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
// CHECK:        arith.muli {{.*}}%[[OFF1]]{{.*}} : index
// CHECK:        arith.addi %[[OFFSET]]{{.*}} : index
// CHECK:        %[[BASE_OFF:.+]] = arith.addi {{.*}}%[[OFF2]]{{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast %[[BASE_OFF]] : index to vector<8xindex>
// CHECK:        %[[LIN:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[BASE_IDX:.+]] = memref.extract_aligned_pointer_as_index %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE_IDX]] : index to i64
// CHECK:        xegpu.store %[[VALS]], %[[BASE_I64]]{{\[}}%[[LIN]]{{\]}}, %[[MASK]] : vector<8xf16>, i64, vector<8xindex>, vector<8xi1>
// CHECK:        gpu.return
}
