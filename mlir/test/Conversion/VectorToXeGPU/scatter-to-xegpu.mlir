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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// CHECK:        xegpu.store %[[VAL]], %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8xf32>, memref<4096xf32>, vector<8xindex>, vector<8xi1>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]{{\]}} : memref<8x32xf32> into memref<256xf32>
// CHECK:        xegpu.store %[[VAL]], %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8xf32>, memref<256xf32>, vector<8xindex>, vector<8xi1>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// CHECK:        xegpu.store %[[VAL]], %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8x16xf32>, memref<4096xf32>, vector<8x16xindex>, vector<8x16xi1>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<?x?x?xf32> into memref<?xf32>
// CHECK:        xegpu.store %[[VAL]], %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8x16xf32>, memref<?xf32>, vector<8x16xindex>, vector<8x16xi1>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<?x8x16xf32> into memref<?xf32>
// CHECK:        xegpu.store %[[VAL]], %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8x16xf32>, memref<?xf32>, vector<8x16xindex>, vector<8x16xi1>
// CHECK:        gpu.return
}

// -----
gpu.module @xevm_module {
gpu.func @no_store_non_unit_inner_stride(
    %vec: vector<8xf32>, %source: memref<32xf32, strided<[?], offset: ?>>,
    %off: index, %indices: vector<8xindex>, %mask: vector<8xi1>) {
  vector.scatter %source[%off][%indices], %mask, %vec
    : memref<32xf32, strided<[?], offset: ?>>, vector<8xindex>, vector<8xi1>, vector<8xf32>
  gpu.return
}
// CHECK-LABEL:  @no_store_non_unit_inner_stride(
// CHECK:        vector.scatter
}

// -----
gpu.module @xevm_module {
gpu.func @store_1D_aligned(%vec: vector<8xf32>, %source: memref<8x16x32xf32>,
     %off1: index, %off2: index, %off3: index,
     %indices: vector<8xindex>, %mask: vector<8xi1>) {
  vector.scatter %source[%off1, %off2, %off3][%indices], %mask, %vec {alignment = 256}
       : memref<8x16x32xf32>, vector<8xindex>, vector<8xi1>, vector<8xf32>
  gpu.return
}
// CHECK-LABEL:  @store_1D_aligned(
// CHECK-SAME:   %[[VAL:.+]]: vector<8xf32>, %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>, %[[MASK:.+]]: vector<8xi1>) {
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// CHECK:        %[[COLLAPSE_ALIGN:.+]] = memref.assume_alignment %[[COLLAPSE]], 256 : memref<4096xf32>
// CHECK:        xegpu.store %[[VAL]], %[[COLLAPSE_ALIGN]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : vector<8xf32>, memref<4096xf32>, vector<8xindex>, vector<8xi1>
// CHECK:        gpu.return
}
