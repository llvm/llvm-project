// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

gpu.module @xevm_module {
gpu.func @load_1D_vector(%source: memref<8x16x32xf32>,
     %off1: index, %off2: index, %off3: index,
     %indices: vector<8xindex>, %mask: vector<8xi1>,
     %pass_thru: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.gather %source[%off1, %off2, %off3][%indices], %mask,
       %pass_thru : memref<8x16x32xf32>, vector<8xindex>, vector<8xi1>, vector<8xf32> into vector<8xf32>
  gpu.return %0 : vector<8xf32>
}
// CHECK-LABEL:  @load_1D_vector(
// CHECK-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>
// CHECK-SAME:   %[[MASK:.+]]: vector<8xi1>
// CHECK-SAME:   %[[PASS_THRU:.+]]: vector<8xf32>) -> vector<8xf32> {
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// CHECK:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : i64, vector<8xindex>, vector<8xi1> -> vector<8xf32>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]] : vector<8xi1>, vector<8xf32>
// CHECK:        gpu.return %[[RES]] : vector<8xf32>
}

// -----
gpu.module @xevm_module {
gpu.func @load_2D_memref(%source: memref<8x32xf32>,
     %off1: index, %off2: index,
     %indices: vector<8xindex>, %mask: vector<8xi1>,
     %pass_thru: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.gather %source[%off1, %off2][%indices], %mask,
       %pass_thru : memref<8x32xf32>, vector<8xindex>, vector<8xi1>, vector<8xf32> into vector<8xf32>
  gpu.return %0 : vector<8xf32>
}
// CHECK-LABEL:  @load_2D_memref(
// CHECK-SAME:   %[[SRC:.+]]: memref<8x32xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>
// CHECK-SAME:   %[[MASK:.+]]: vector<8xi1>
// CHECK-SAME:   %[[PASS_THRU:.+]]: vector<8xf32>) -> vector<8xf32> {
// CHECK-COUNT1: arith.muli {{.*}} : index
// CHECK-COUNT1: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x32xf32> -> index
// CHECK:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : i64, vector<8xindex>, vector<8xi1> -> vector<8xf32>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]] : vector<8xi1>, vector<8xf32>
// CHECK:        gpu.return %[[RES]] : vector<8xf32>
}

// -----
gpu.module @xevm_module {
gpu.func @load_2D_vector(%source: memref<8x16x32xf32>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8x16xindex>, %mask: vector<8x16xi1>,
    %pass_thru: vector<8x16xf32>) -> vector<8x16xf32> {
  %0 = vector.gather %source[%off0, %off1, %off2][%indices], %mask,
       %pass_thru : memref<8x16x32xf32>, vector<8x16xindex>, vector<8x16xi1>, vector<8x16xf32> into vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}
// CHECK-LABEL:  @load_2D_vector(
// CHECK-SAME:   %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8x16xindex>
// CHECK-SAME:   %[[MASK:.+]]: vector<8x16xi1>
// CHECK-SAME:   %[[PASS_THRU:.+]]: vector<8x16xf32>) -> vector<8x16xf32> {
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8x16xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8x16xindex>
// CHECK:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<8x16x32xf32> -> index
// CHECK:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]] : vector<8x16xi1>, vector<8x16xf32>
// CHECK:        gpu.return %[[RES]] : vector<8x16xf32>
}

// -----
gpu.module @xevm_module {
gpu.func @load_dynamic_source(%source: memref<?x?x?xf32>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8x16xindex>, %mask: vector<8x16xi1>,
    %pass_thru: vector<8x16xf32>) -> vector<8x16xf32> {
  %0 = vector.gather %source[%off0, %off1, %off2][%indices], %mask,
       %pass_thru : memref<?x?x?xf32>, vector<8x16xindex>, vector<8x16xi1>, vector<8x16xf32> into vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}
// CHECK-LABEL:  @load_dynamic_source(
// CHECK-SAME:   %[[SRC:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8x16xindex>
// CHECK-SAME:   %[[MASK:.+]]: vector<8x16xi1>
// CHECK-SAME:   %[[PASS_THRU:.+]]: vector<8x16xf32>) -> vector<8x16xf32> {
// CHECK:        memref.extract_strided_metadata %[[SRC]]
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8x16xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8x16xindex>
// CHECK:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<?x?x?xf32> -> index
// CHECK:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]] : vector<8x16xi1>, vector<8x16xf32>
// CHECK:        gpu.return %[[RES]] : vector<8x16xf32>
}

// -----
gpu.module @xevm_module {
gpu.func @load_dynamic_source2(%source: memref<?x8x16xf32>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8x16xindex>, %mask: vector<8x16xi1>,
    %pass_thru: vector<8x16xf32>) -> vector<8x16xf32> {
  %0 = vector.gather %source[%off0, %off1, %off2][%indices], %mask,
       %pass_thru : memref<?x8x16xf32>, vector<8x16xindex>, vector<8x16xi1>, vector<8x16xf32> into vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}
// CHECK-LABEL:  @load_dynamic_source2(
// CHECK-SAME:   %[[SRC:.+]]: memref<?x8x16xf32>,
// CHECK-SAME:   %[[OFF1:.+]]: index, %[[OFF2:.+]]: index, %[[OFF3:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8x16xindex>
// CHECK-SAME:   %[[MASK:.+]]: vector<8x16xi1>
// CHECK-SAME:   %[[PASS_THRU:.+]]: vector<8x16xf32>) -> vector<8x16xf32> {
// CHECK-NOT:    memref.extract_strided_metadata %[[SRC]]
// CHECK-COUNT2: arith.muli {{.*}} : index
// CHECK-COUNT2: arith.addi {{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8x16xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8x16xindex>
// CHECK:        %[[COLLAPSE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<?x8x16xf32> -> index
// CHECK:        %[[COLLAPSE_I:.+]] = arith.index_cast %[[COLLAPSE]] : index to i64
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE_I]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]] : vector<8x16xi1>, vector<8x16xf32>
// CHECK:        gpu.return %[[RES]] : vector<8x16xf32>
}

// -----
gpu.module @xevm_module {
gpu.func @no_load_tensor(%source: tensor<32x64xf32>,
    %off: index, %indices: vector<8x16xindex>,
    %mask: vector<8x16xi1>, %pass_thru: vector<8x16xf32>) -> vector<8x16xf32> {
  %0 = vector.gather %source[%off, %off][%indices], %mask,
       %pass_thru : tensor<32x64xf32>, vector<8x16xindex>, vector<8x16xi1>, vector<8x16xf32> into vector<8x16xf32>
  gpu.return %0 : vector<8x16xf32>
}
// CHECK-LABEL:  @no_load_tensor(
// CHECK:        vector.gather
}

// -----
gpu.module @xevm_module {
gpu.func @gather_from_subview(%source: memref<4096x4096xf16>,
                              %memref_off: index, %off1: index, %off2: index,
                              %indices: vector<8xindex>,
                              %mask: vector<8xi1>,
                              %pass_thru: vector<8xf16>) -> vector<8xf16> {
  %subview = memref.subview %source[%memref_off, %memref_off] [256, 256] [1, 1]
      : memref<4096x4096xf16>
        to memref<256x256xf16, strided<[4096, 1], offset: ?>>
  %0 = vector.gather %subview[%off1, %off2][%indices], %mask, %pass_thru
       : memref<256x256xf16, strided<[4096, 1], offset: ?>>,
         vector<8xindex>, vector<8xi1>, vector<8xf16>
         into vector<8xf16>
  gpu.return %0 : vector<8xf16>
}
// CHECK-LABEL:  @gather_from_subview(
// CHECK-SAME:   %[[SRC:.+]]: memref<4096x4096xf16>,
// CHECK-SAME:   %[[MEMREF_OFF:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>,
// CHECK-SAME:   %[[MASK:.+]]: vector<8xi1>,
// CHECK-SAME:   %[[PASS:.+]]: vector<8xf16>) -> vector<8xf16> {
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[SRC]][%[[MEMREF_OFF]], %[[MEMREF_OFF]]] [256, 256] [1, 1]
// CHECK:        %[[BB:.+]], %[[OFFSET:.+]],{{.*}},{{.*}} = memref.extract_strided_metadata %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
// CHECK:        arith.muli {{.*}}%[[OFF1]]{{.*}} : index
// CHECK:        arith.addi %[[OFFSET]]{{.*}} : index
// CHECK:        %[[BASE_OFF:.+]] = arith.addi {{.*}}%[[OFF2]]{{.*}} : index
// CHECK:        %[[SPLAT:.+]] = vector.broadcast %[[BASE_OFF]] : index to vector<8xindex>
// CHECK:        %[[LIN:.+]] = arith.addi %[[SPLAT]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[BASE_IDX:.+]] = memref.extract_aligned_pointer_as_index %[[SUBVIEW]] : memref<256x256xf16, strided<[4096, 1], offset: ?>> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE_IDX]] : index to i64
// CHECK:        %[[VEC:.+]] = xegpu.load %[[BASE_I64]]{{\[}}%[[LIN]]{{\]}}, %[[MASK]]
// CHECK-SAME:     : i64, vector<8xindex>, vector<8xi1> -> vector<8xf16>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS]] : vector<8xi1>, vector<8xf16>
// CHECK:        gpu.return %[[RES]] : vector<8xf16>
}

// -----
gpu.module @xevm_module {
gpu.func @non_unit_inner_stride_1D(
    %source: memref<32xf32, strided<[?], offset: ?>>,
    %off: index, %indices: vector<8xindex>, %mask: vector<8xi1>,
    %pass_thru: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.gather %source[%off][%indices], %mask, %pass_thru
       : memref<32xf32, strided<[?], offset: ?>>,
         vector<8xindex>, vector<8xi1>, vector<8xf32>
         into vector<8xf32>
  gpu.return %0 : vector<8xf32>
}
// CHECK-LABEL:  @non_unit_inner_stride_1D(
// CHECK-SAME:   %[[SRC:.+]]: memref<32xf32, strided<[?], offset: ?>>,
// CHECK-SAME:   %[[OFF1:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>,
// CHECK-SAME:   %[[MASK:.+]]: vector<8xi1>, %[[PASS:.+]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:        %[[BB:.+]], %[[M_OFF:.+]], %[[SZ:.+]], %[[STRIDE:.+]] = memref.extract_strided_metadata %[[SRC]]
// CHECK:        arith.muli %[[OFF1]], %[[STRIDE]] : index
// CHECK:        arith.addi {{.*}} : index
// CHECK:        %[[STRD_VEC:.+]] = vector.broadcast %[[STRIDE]] : index to vector<8xindex>
// CHECK:        %[[STRD_INDICES:.+]] = arith.muli %[[STRD_VEC:.+]], %[[INDICES]] : vector<8xindex>
// CHECK:        %[[SPLAT:.+]] = vector.broadcast {{.*}}:  index to vector<8xindex>
// CHECK:        %[[LIN_IDX:.+]] = arith.addi %[[SPLAT]], %[[STRD_INDICES]] : vector<8xindex>
// CHECK:        %[[BASE:.+]] = memref.extract_aligned_pointer_as_index %[[SRC]] : memref<32xf32, strided<[?], offset: ?>> -> index
// CHECK:        %[[BASE_I64:.+]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:        %[[V:.+]] = xegpu.load %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : i64, vector<8xindex>, vector<8xi1> -> vector<8xf32>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[V]], %[[PASS]] : vector<8xi1>, vector<8xf32>
// CHECK:        gpu.return %[[RES]] : vector<8xf32>
}

// -----
gpu.module @xevm_module {
gpu.func @non_unit_inner_stride_3D(
    %source: memref<4x8x32xf32, strided<[?, 128, 2], offset: ?>>,
    %off0: index, %off1: index, %off2: index,
    %indices: vector<8xindex>, %mask: vector<8xi1>,
    %pass_thru: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.gather %source[%off0, %off1, %off2][%indices], %mask, %pass_thru
       : memref<4x8x32xf32, strided<[?, 128, 2], offset: ?>>,
         vector<8xindex>, vector<8xi1>, vector<8xf32>
         into vector<8xf32>
  gpu.return %0 : vector<8xf32>
}
// CHECK-LABEL:  @non_unit_inner_stride_3D(
// CHECK-SAME:   %[[SRC:.+]]: memref<4x8x32xf32, strided<[?, 128, 2], offset: ?>>,
// CHECK-SAME:   %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index,
// CHECK-SAME:   %[[INDICES:.+]]: vector<8xindex>, %[[MASK:.+]]: vector<8xi1>,
// CHECK-SAME:   %[[PASS:.+]]: vector<8xf32>) -> vector<8xf32> {
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
// CHECK:        %[[V:.+]] = xegpu.load %[[BASE_I64]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : i64, vector<8xindex>, vector<8xi1> -> vector<8xf32>
// CHECK:        %[[RES:.+]] = arith.select %[[MASK]], %[[V]], %[[PASS]] : vector<8xi1>, vector<8xf32>
// CHECK:        gpu.return %[[RES]] : vector<8xf32>
}
