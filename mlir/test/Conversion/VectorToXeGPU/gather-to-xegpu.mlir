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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : memref<4096xf32>, vector<8xindex>, vector<8xi1> -> vector<8xf32>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1]{{\]}} : memref<8x32xf32> into memref<256xf32>
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : memref<256xf32>, vector<8xindex>, vector<8xi1> -> vector<8xf32>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<8x16x32xf32> into memref<4096xf32>
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : memref<4096xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<?x?x?xf32> into memref<?xf32>
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : memref<?xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
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
// CHECK:        %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC]] {{\[}}[0, 1, 2]{{\]}} : memref<?x8x16xf32> into memref<?xf32>
// CHECK:        %[[VEC:.+]] = xegpu.load %[[COLLAPSE]]{{\[}}%[[LIN_IDX]]{{\]}}, %[[MASK]] : memref<?xf32>, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
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
gpu.func @no_load_non_unit_inner_stride(
    %source: memref<32xf32, strided<[?], offset: ?>>,
    %off: index, %indices: vector<8xindex>, %mask: vector<8xi1>,
    %pass_thru: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.gather %source[%off][%indices], %mask, %pass_thru
    : memref<32xf32, strided<[?], offset: ?>>, vector<8xindex>, vector<8xi1>, vector<8xf32> into vector<8xf32>
  gpu.return %0 : vector<8xf32>
}
// CHECK-LABEL:  @no_load_non_unit_inner_stride(
// CHECK:        vector.gather
}

