// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s --check-prefix=SHUFFLE

!memrefA = memref<32x128x128xbf16>
!memrefB = memref<32x128x128xbf16>
!memrefC = memref<128x128xf32>
!subA = memref<1x32x8xbf16, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x8xbf16>
!vecB = vector<1x8x16xbf16>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_bf16_case3(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c8
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10]
              [1, 32, 8] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4]
              [1, 8, 32] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK:         memref.global "private" constant @mask_1 : memref<16xi32> = dense<[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK:         memref.global "private" constant @mask_2 : memref<16xi32> = dense<[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]>
// CHECK:         memref.global "private" constant @mask_3 : memref<16xi32> = dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]>

// CHECK-LABEL:   func.func @brgemm_bf16_case3(
// CHECK-SAME:      %[[ARG0:[a-z0-9]+]]: memref<32x128x128xbf16>,
// CHECK-SAME:      %[[ARG1:[a-z0-9]+]]: memref<32x128x128xbf16>,
// CHECK-SAME:      %[[ARG2:[a-z0-9]+]]: memref<128x128xf32>) {
// CHECK:           %[[POISON_0:.*]] = ub.poison : f32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : i16
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 16 : i16
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 39 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 38 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 7 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 6 : index
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 37 : index
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 36 : index
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 5 : index
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_10:.*]] = arith.constant 35 : index
// CHECK:           %[[CONSTANT_11:.*]] = arith.constant 34 : index
// CHECK:           %[[CONSTANT_12:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_13:.*]] = arith.constant 33 : index
// CHECK:           %[[CONSTANT_14:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_15:.*]] = arith.constant dense<1> : vector<16xi32>
// CHECK:           %[[CONSTANT_16:.*]] = arith.constant 16 : index
// CHECK:           %[[CONSTANT_17:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_18:.*]] = arith.constant 128 : index
// CHECK:           %[[CONSTANT_19:.*]] = arith.constant 32 : index
// CHECK:           %[[CONSTANT_20:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_21:.*]] = arith.constant 8 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_17]] to %[[CONSTANT_18]] step %[[CONSTANT_19]] {
// CHECK:             scf.for %[[VAL_1:.*]] = %[[CONSTANT_17]] to %[[CONSTANT_18]] step %[[CONSTANT_19]] {
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ARG2]]{{\[}}%[[VAL_0]], %[[VAL_1]]] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32, strided<[128, 1], offset: ?>>
// CHECK-NOT:           x86.ace.bsr_init
// CHECK:               %[[VAL_2:.*]] = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK:               %[[VAL_3:.*]] = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK:               %[[VAL_4:.*]] = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK:               %[[VAL_5:.*]] = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK:               %[[GET_GLOBAL_0:.*]] = memref.get_global @mask_1 : memref<16xi32>
// CHECK:               %[[LOAD_0:.*]] = vector.load %[[GET_GLOBAL_0]]{{\[}}%[[CONSTANT_17]]] : memref<16xi32>, vector<16xi32>
// CHECK:               %[[GET_GLOBAL_1:.*]] = memref.get_global @mask_2 : memref<16xi32>
// CHECK:               %[[LOAD_1:.*]] = vector.load %[[GET_GLOBAL_1]]{{\[}}%[[CONSTANT_17]]] : memref<16xi32>, vector<16xi32>
// CHECK:               %[[GET_GLOBAL_2:.*]] = memref.get_global @mask_3 : memref<16xi32>
// CHECK:               %[[LOAD_2:.*]] = vector.load %[[GET_GLOBAL_2]]{{\[}}%[[CONSTANT_17]]] : memref<16xi32>, vector<16xi32>
// CHECK:               %[[FOR_0:.*]]:4 = scf.for %[[VAL_6:.*]] = %[[CONSTANT_17]] to %[[CONSTANT_19]] step %[[CONSTANT_20]] iter_args(%[[VAL_7:.*]] = %[[VAL_2]], %[[VAL_8:.*]] = %[[VAL_3]], %[[VAL_9:.*]] = %[[VAL_4]], %[[VAL_10:.*]] = %[[VAL_5]]) -> (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>) {
// CHECK:                 %[[FOR_1:.*]]:4 = scf.for %[[VAL_11:.*]] = %[[CONSTANT_17]] to %[[CONSTANT_18]] step %[[CONSTANT_21]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]], %[[VAL_13:.*]] = %[[VAL_8]], %[[VAL_14:.*]] = %[[VAL_9]], %[[VAL_15:.*]] = %[[VAL_10]]) -> (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>) {
// CHECK:                   %[[SUBVIEW_1:.*]] = memref.subview %[[ARG1]]{{\[}}%[[VAL_6]], %[[VAL_11]], %[[VAL_1]]] [1, 8, 32] [1, 1, 1] : memref<32x128x128xbf16> to memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<32x128x128xbf16> to !llvm.ptr
// CHECK:                   %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[UNREALIZED_CONVERSION_CAST_0]] : !llvm.ptr to memref<32x128x64xi32>
// CHECK:                   %[[DIVUI_0:.*]] = arith.divui %[[VAL_11]], %[[CONSTANT_14]] : index
// CHECK:                   %[[LOAD_3:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[VAL_0]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_4:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_0]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_5:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_1]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_6:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_2]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[CMPI_0:.*]] = arith.cmpi eq, %[[LOAD_0]], %[[CONSTANT_15]] : vector<16xi32>
// CHECK:                   %[[ADDI_3:.*]] = arith.addi %[[ADDI_2]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_7:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_3]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_0:.*]] = vector.shuffle %[[LOAD_7]], %[[LOAD_7]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_0]], %[[LOAD_3]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_4:.*]] = arith.addi %[[ADDI_3]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_8:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_4]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_1:.*]] = vector.shuffle %[[LOAD_8]], %[[LOAD_8]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_1]], %[[LOAD_4]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_5:.*]] = arith.addi %[[ADDI_4]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_9:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_5]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_2:.*]] = vector.shuffle %[[LOAD_9]], %[[LOAD_9]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_2:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_2]], %[[LOAD_5]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_6:.*]] = arith.addi %[[ADDI_5]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_10:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_6]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_3:.*]] = vector.shuffle %[[LOAD_10]], %[[LOAD_10]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_3:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_3]], %[[LOAD_6]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[CMPI_1:.*]] = arith.cmpi eq, %[[LOAD_1]], %[[CONSTANT_15]] : vector<16xi32>
// CHECK:                   %[[ADDI_7:.*]] = arith.addi %[[ADDI_6]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_11:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_7]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_4:.*]] = vector.shuffle %[[LOAD_11]], %[[LOAD_11]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_4:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_4]], %[[SELECT_0]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_8:.*]] = arith.addi %[[ADDI_7]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_12:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_8]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_5:.*]] = vector.shuffle %[[LOAD_12]], %[[LOAD_12]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_5:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_5]], %[[SELECT_1]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_9:.*]] = arith.addi %[[ADDI_8]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_13:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_9]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_6:.*]] = vector.shuffle %[[LOAD_13]], %[[LOAD_13]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_6:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_6]], %[[SELECT_2]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_10:.*]] = arith.addi %[[ADDI_9]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_14:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_10]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_7:.*]] = vector.shuffle %[[LOAD_14]], %[[LOAD_14]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_7:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_7]], %[[SELECT_3]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[CMPI_2:.*]] = arith.cmpi eq, %[[LOAD_2]], %[[CONSTANT_15]] : vector<16xi32>
// CHECK:                   %[[ADDI_11:.*]] = arith.addi %[[ADDI_10]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_15:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_11]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_8:.*]] = vector.shuffle %[[LOAD_15]], %[[LOAD_15]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_8:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_8]], %[[SELECT_4]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_12:.*]] = arith.addi %[[ADDI_11]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_16:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_12]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_9:.*]] = vector.shuffle %[[LOAD_16]], %[[LOAD_16]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_9:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_9]], %[[SELECT_5]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_13:.*]] = arith.addi %[[ADDI_12]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_17:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_13]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_10:.*]] = vector.shuffle %[[LOAD_17]], %[[LOAD_17]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_10:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_10]], %[[SELECT_6]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_14:.*]] = arith.addi %[[ADDI_13]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_18:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_14]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_11:.*]] = vector.shuffle %[[LOAD_18]], %[[LOAD_18]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_11:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_11]], %[[SELECT_7]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[BITCAST_0:.*]] = vector.bitcast %[[SELECT_8]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[BITCAST_1:.*]] = vector.bitcast %[[SELECT_9]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[BITCAST_2:.*]] = vector.bitcast %[[SELECT_10]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[BITCAST_3:.*]] = vector.bitcast %[[SELECT_11]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[SHUFFLE_12:.*]] = vector.shuffle %[[BITCAST_0]], %[[BITCAST_1]] [0, 1, 32, 33, 2, 3, 34, 35, 8, 9, 40, 41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19, 50, 51, 24, 25, 56, 57, 26, 27, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_13:.*]] = vector.shuffle %[[BITCAST_0]], %[[BITCAST_1]] [4, 5, 36, 37, 6, 7, 38, 39, 12, 13, 44, 45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23, 54, 55, 28, 29, 60, 61, 30, 31, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_14:.*]] = vector.shuffle %[[BITCAST_2]], %[[BITCAST_3]] [0, 1, 32, 33, 2, 3, 34, 35, 8, 9, 40, 41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19, 50, 51, 24, 25, 56, 57, 26, 27, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_15:.*]] = vector.shuffle %[[BITCAST_2]], %[[BITCAST_3]] [4, 5, 36, 37, 6, 7, 38, 39, 12, 13, 44, 45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23, 54, 55, 28, 29, 60, 61, 30, 31, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_16:.*]] = vector.shuffle %[[SHUFFLE_12]], %[[SHUFFLE_14]] [0, 1, 2, 3, 32, 33, 34, 35, 8, 9, 10, 11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49, 50, 51, 24, 25, 26, 27, 56, 57, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_17:.*]] = vector.shuffle %[[SHUFFLE_12]], %[[SHUFFLE_14]] [4, 5, 6, 7, 36, 37, 38, 39, 12, 13, 14, 15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53, 54, 55, 28, 29, 30, 31, 60, 61, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_18:.*]] = vector.shuffle %[[SHUFFLE_13]], %[[SHUFFLE_15]] [0, 1, 2, 3, 32, 33, 34, 35, 8, 9, 10, 11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49, 50, 51, 24, 25, 26, 27, 56, 57, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_19:.*]] = vector.shuffle %[[SHUFFLE_13]], %[[SHUFFLE_15]] [4, 5, 6, 7, 36, 37, 38, 39, 12, 13, 14, 15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53, 54, 55, 28, 29, 30, 31, 60, 61, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[ADDI_15:.*]] = arith.addi %[[ADDI_14]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_19:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_15]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[ADDI_16:.*]] = arith.addi %[[ADDI_15]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_20:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_16]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[ADDI_17:.*]] = arith.addi %[[ADDI_16]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_21:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_17]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[ADDI_18:.*]] = arith.addi %[[ADDI_17]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_22:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_18]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<16xi32>
// CHECK:                   %[[ADDI_19:.*]] = arith.addi %[[ADDI_18]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_23:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_19]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_20:.*]] = vector.shuffle %[[LOAD_23]], %[[LOAD_23]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_12:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_20]], %[[LOAD_19]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_20:.*]] = arith.addi %[[ADDI_19]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_24:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_20]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_21:.*]] = vector.shuffle %[[LOAD_24]], %[[LOAD_24]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_13:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_21]], %[[LOAD_20]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_21:.*]] = arith.addi %[[ADDI_20]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_25:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_21]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_22:.*]] = vector.shuffle %[[LOAD_25]], %[[LOAD_25]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_14:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_22]], %[[LOAD_21]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_22:.*]] = arith.addi %[[ADDI_21]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_26:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_22]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_23:.*]] = vector.shuffle %[[LOAD_26]], %[[LOAD_26]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_15:.*]] = arith.select %[[CMPI_0]], %[[SHUFFLE_23]], %[[LOAD_22]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_23:.*]] = arith.addi %[[ADDI_22]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_27:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_23]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_24:.*]] = vector.shuffle %[[LOAD_27]], %[[LOAD_27]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_16:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_24]], %[[SELECT_12]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_24:.*]] = arith.addi %[[ADDI_23]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_28:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_24]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_25:.*]] = vector.shuffle %[[LOAD_28]], %[[LOAD_28]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_17:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_25]], %[[SELECT_13]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_25:.*]] = arith.addi %[[ADDI_24]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_29:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_25]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_26:.*]] = vector.shuffle %[[LOAD_29]], %[[LOAD_29]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_18:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_26]], %[[SELECT_14]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_26:.*]] = arith.addi %[[ADDI_25]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_30:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_26]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_27:.*]] = vector.shuffle %[[LOAD_30]], %[[LOAD_30]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_19:.*]] = arith.select %[[CMPI_1]], %[[SHUFFLE_27]], %[[SELECT_15]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_27:.*]] = arith.addi %[[ADDI_26]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_31:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_27]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_28:.*]] = vector.shuffle %[[LOAD_31]], %[[LOAD_31]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_20:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_28]], %[[SELECT_16]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_28:.*]] = arith.addi %[[ADDI_27]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_32:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_28]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_29:.*]] = vector.shuffle %[[LOAD_32]], %[[LOAD_32]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_21:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_29]], %[[SELECT_17]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_29:.*]] = arith.addi %[[ADDI_28]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_33:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_29]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_30:.*]] = vector.shuffle %[[LOAD_33]], %[[LOAD_33]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_22:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_30]], %[[SELECT_18]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[ADDI_30:.*]] = arith.addi %[[ADDI_29]], %[[CONSTANT_20]] : index
// CHECK:                   %[[LOAD_34:.*]] = vector.load %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_6]], %[[ADDI_30]], %[[DIVUI_0]]] : memref<32x128x64xi32>, vector<4xi32>
// CHECK:                   %[[SHUFFLE_31:.*]] = vector.shuffle %[[LOAD_34]], %[[LOAD_34]] [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi32>, vector<4xi32>
// CHECK:                   %[[SELECT_23:.*]] = arith.select %[[CMPI_2]], %[[SHUFFLE_31]], %[[SELECT_19]] : vector<16xi1>, vector<16xi32>
// CHECK:                   %[[BITCAST_4:.*]] = vector.bitcast %[[SELECT_20]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[BITCAST_5:.*]] = vector.bitcast %[[SELECT_21]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[BITCAST_6:.*]] = vector.bitcast %[[SELECT_22]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[BITCAST_7:.*]] = vector.bitcast %[[SELECT_23]] : vector<16xi32> to vector<32xbf16>
// CHECK:                   %[[SHUFFLE_32:.*]] = vector.shuffle %[[BITCAST_4]], %[[BITCAST_5]] [0, 1, 32, 33, 2, 3, 34, 35, 8, 9, 40, 41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19, 50, 51, 24, 25, 56, 57, 26, 27, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_33:.*]] = vector.shuffle %[[BITCAST_4]], %[[BITCAST_5]] [4, 5, 36, 37, 6, 7, 38, 39, 12, 13, 44, 45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23, 54, 55, 28, 29, 60, 61, 30, 31, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_34:.*]] = vector.shuffle %[[BITCAST_6]], %[[BITCAST_7]] [0, 1, 32, 33, 2, 3, 34, 35, 8, 9, 40, 41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19, 50, 51, 24, 25, 56, 57, 26, 27, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_35:.*]] = vector.shuffle %[[BITCAST_6]], %[[BITCAST_7]] [4, 5, 36, 37, 6, 7, 38, 39, 12, 13, 44, 45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23, 54, 55, 28, 29, 60, 61, 30, 31, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_36:.*]] = vector.shuffle %[[SHUFFLE_32]], %[[SHUFFLE_34]] [0, 1, 2, 3, 32, 33, 34, 35, 8, 9, 10, 11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49, 50, 51, 24, 25, 26, 27, 56, 57, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_37:.*]] = vector.shuffle %[[SHUFFLE_32]], %[[SHUFFLE_34]] [4, 5, 6, 7, 36, 37, 38, 39, 12, 13, 14, 15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53, 54, 55, 28, 29, 30, 31, 60, 61, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_38:.*]] = vector.shuffle %[[SHUFFLE_33]], %[[SHUFFLE_35]] [0, 1, 2, 3, 32, 33, 34, 35, 8, 9, 10, 11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49, 50, 51, 24, 25, 26, 27, 56, 57, 58, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_39:.*]] = vector.shuffle %[[SHUFFLE_33]], %[[SHUFFLE_35]] [4, 5, 6, 7, 36, 37, 38, 39, 12, 13, 14, 15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53, 54, 55, 28, 29, 30, 31, 60, 61, 62, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[LOAD_35:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_17]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_19]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[LOAD_36:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_20]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_13]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[SHUFFLE_40:.*]] = vector.shuffle %[[LOAD_35]], %[[LOAD_36]] [0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_41:.*]] = vector.shuffle %[[LOAD_35]], %[[LOAD_36]] [4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[VAL_16:.*]] = x86.ace.tile_mulf %[[SHUFFLE_16]], %[[SHUFFLE_40]], %[[VAL_12]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_17:.*]] = x86.ace.tile_mulf %[[SHUFFLE_16]], %[[SHUFFLE_41]], %[[VAL_13]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_18:.*]] = x86.ace.tile_mulf %[[SHUFFLE_36]], %[[SHUFFLE_40]], %[[VAL_14]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_19:.*]] = x86.ace.tile_mulf %[[SHUFFLE_36]], %[[SHUFFLE_41]], %[[VAL_15]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[LOAD_37:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_14]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_11]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[LOAD_38:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_12]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_10]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[SHUFFLE_42:.*]] = vector.shuffle %[[LOAD_37]], %[[LOAD_38]] [0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_43:.*]] = vector.shuffle %[[LOAD_37]], %[[LOAD_38]] [4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[VAL_20:.*]] = x86.ace.tile_mulf %[[SHUFFLE_17]], %[[SHUFFLE_42]], %[[VAL_16]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_21:.*]] = x86.ace.tile_mulf %[[SHUFFLE_17]], %[[SHUFFLE_43]], %[[VAL_17]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_22:.*]] = x86.ace.tile_mulf %[[SHUFFLE_37]], %[[SHUFFLE_42]], %[[VAL_18]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_23:.*]] = x86.ace.tile_mulf %[[SHUFFLE_37]], %[[SHUFFLE_43]], %[[VAL_19]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[LOAD_39:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_9]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_7]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[LOAD_40:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_8]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_6]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[SHUFFLE_44:.*]] = vector.shuffle %[[LOAD_39]], %[[LOAD_40]] [0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_45:.*]] = vector.shuffle %[[LOAD_39]], %[[LOAD_40]] [4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[VAL_24:.*]] = x86.ace.tile_mulf %[[SHUFFLE_18]], %[[SHUFFLE_44]], %[[VAL_20]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_25:.*]] = x86.ace.tile_mulf %[[SHUFFLE_18]], %[[SHUFFLE_45]], %[[VAL_21]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_26:.*]] = x86.ace.tile_mulf %[[SHUFFLE_38]], %[[SHUFFLE_44]], %[[VAL_22]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_27:.*]] = x86.ace.tile_mulf %[[SHUFFLE_38]], %[[SHUFFLE_45]], %[[VAL_23]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[LOAD_41:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_5]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_3]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[LOAD_42:.*]] = vector.load %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_4]], %[[CONSTANT_17]]] : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>, vector<32xbf16>
// CHECK:                   memref.prefetch %[[SUBVIEW_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_2]], %[[CONSTANT_17]]], read, locality<3>, data : memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
// CHECK:                   %[[SHUFFLE_46:.*]] = vector.shuffle %[[LOAD_41]], %[[LOAD_42]] [0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[SHUFFLE_47:.*]] = vector.shuffle %[[LOAD_41]], %[[LOAD_42]] [4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK:                   %[[VAL_28:.*]] = x86.ace.tile_mulf %[[SHUFFLE_19]], %[[SHUFFLE_46]], %[[VAL_24]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_29:.*]] = x86.ace.tile_mulf %[[SHUFFLE_19]], %[[SHUFFLE_47]], %[[VAL_25]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_30:.*]] = x86.ace.tile_mulf %[[SHUFFLE_39]], %[[SHUFFLE_46]], %[[VAL_26]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   %[[VAL_31:.*]] = x86.ace.tile_mulf %[[SHUFFLE_39]], %[[SHUFFLE_47]], %[[VAL_27]] : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK:                   scf.yield %[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_31]] : !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[FOR_1]]#0, %[[FOR_1]]#1, %[[FOR_1]]#2, %[[FOR_1]]#3 : !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>
// CHECK:               }
// CHECK:               %[[SUBVIEW_2:.*]] = memref.subview %[[SUBVIEW_0]][0, 0] [16, 16] [1, 1] : memref<32x32xf32, strided<[128, 1], offset: ?>> to memref<16x16xf32, strided<[128, 1], offset: ?>>
// CHECK:               %[[SUBVIEW_3:.*]] = memref.subview %[[SUBVIEW_0]][0, 16] [16, 16] [1, 1] : memref<32x32xf32, strided<[128, 1], offset: ?>> to memref<16x16xf32, strided<[128, 1], offset: ?>>
// CHECK:               %[[ALLOCA_0:.*]] = memref.alloca() : memref<16x16xf32>
// CHECK:               %[[ALLOCA_1:.*]] = memref.alloca() : memref<16x16xf32>
// CHECK:               %[[SUBVIEW_4:.*]] = memref.subview %[[SUBVIEW_0]][16, 0] [16, 16] [1, 1] : memref<32x32xf32, strided<[128, 1], offset: ?>> to memref<16x16xf32, strided<[128, 1], offset: ?>>
// CHECK:               %[[SUBVIEW_5:.*]] = memref.subview %[[SUBVIEW_0]][16, 16] [16, 16] [1, 1] : memref<32x32xf32, strided<[128, 1], offset: ?>> to memref<16x16xf32, strided<[128, 1], offset: ?>>
// CHECK:               %[[ALLOCA_2:.*]] = memref.alloca() : memref<16x16xf32>
// CHECK:               %[[ALLOCA_3:.*]] = memref.alloca() : memref<16x16xf32>
// CHECK:               scf.for %[[VAL_33:.*]] = %[[CONSTANT_17]] to %[[CONSTANT_16]] step %[[CONSTANT_20]] {
// CHECK:                 %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_33]] : index to i32
// CHECK:                 %[[VAL_34:.*]] = x86.amx.tilemovrow %[[CONSTANT_1]], %[[CONSTANT_0]], %[[FOR_0]]#0, %[[INDEX_CAST_0]] : (i16, i16, !x86.amx.tile<16x16xf32>, i32) -> vector<16xi32>
// CHECK:                 %[[BITCAST_8:.*]] = vector.bitcast %[[VAL_34]] : vector<16xi32> to vector<16xf32>
// CHECK:                 %[[VAL_36:.*]] = x86.amx.tilemovrow %[[CONSTANT_1]], %[[CONSTANT_0]], %[[FOR_0]]#1, %[[INDEX_CAST_0]] : (i16, i16, !x86.amx.tile<16x16xf32>, i32) -> vector<16xi32>
// CHECK:                 %[[BITCAST_9:.*]] = vector.bitcast %[[VAL_36]] : vector<16xi32> to vector<16xf32>
// CHECK:                 %[[SHUFFLE_48:.*]] = vector.shuffle %[[BITCAST_8]], %[[BITCAST_9]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[SHUFFLE_49:.*]] = vector.shuffle %[[BITCAST_8]], %[[BITCAST_9]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[SHUFFLE_50:.*]] = vector.shuffle %[[SHUFFLE_48]], %[[SHUFFLE_48]] [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[SHUFFLE_51:.*]] = vector.shuffle %[[SHUFFLE_49]], %[[SHUFFLE_49]] [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[LOAD_43:.*]] = vector.load %[[SUBVIEW_2]]{{\[}}%[[VAL_33]], %[[CONSTANT_17]]] : memref<16x16xf32, strided<[128, 1], offset: ?>>, vector<16xf32>
// CHECK:                 %[[ADDF_0:.*]] = arith.addf %[[SHUFFLE_50]], %[[LOAD_43]] : vector<16xf32>
// CHECK:                 vector.store %[[ADDF_0]], %[[ALLOCA_0]]{{\[}}%[[VAL_33]], %[[CONSTANT_17]]] : memref<16x16xf32>, vector<16xf32>
// CHECK:                 %[[LOAD_44:.*]] = vector.load %[[SUBVIEW_3]]{{\[}}%[[VAL_33]], %[[CONSTANT_17]]] : memref<16x16xf32, strided<[128, 1], offset: ?>>, vector<16xf32>
// CHECK:                 %[[ADDF_1:.*]] = arith.addf %[[SHUFFLE_51]], %[[LOAD_44]] : vector<16xf32>
// CHECK:                 vector.store %[[ADDF_1]], %[[ALLOCA_1]]{{\[}}%[[VAL_33]], %[[CONSTANT_17]]] : memref<16x16xf32>, vector<16xf32>
// CHECK:               }
// CHECK:               %[[TRANSFER_READ_0:.*]] = vector.transfer_read %[[ALLOCA_0]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_17]]], %[[POISON_0]] {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
// CHECK:               %[[TRANSFER_READ_1:.*]] = vector.transfer_read %[[ALLOCA_1]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_17]]], %[[POISON_0]] {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
// CHECK:               scf.for %[[VAL_37:.*]] = %[[CONSTANT_17]] to %[[CONSTANT_16]] step %[[CONSTANT_20]] {
// CHECK:                 %[[INDEX_CAST_1:.*]] = arith.index_cast %[[VAL_37]] : index to i32
// CHECK:                 %[[VAL_38:.*]] = x86.amx.tilemovrow %[[CONSTANT_1]], %[[CONSTANT_0]], %[[FOR_0]]#2, %[[INDEX_CAST_1]] : (i16, i16, !x86.amx.tile<16x16xf32>, i32) -> vector<16xi32>
// CHECK:                 %[[BITCAST_10:.*]] = vector.bitcast %[[VAL_38]] : vector<16xi32> to vector<16xf32>
// CHECK:                 %[[VAL_40:.*]] = x86.amx.tilemovrow %[[CONSTANT_1]], %[[CONSTANT_0]], %[[FOR_0]]#3, %[[INDEX_CAST_1]] : (i16, i16, !x86.amx.tile<16x16xf32>, i32) -> vector<16xi32>
// CHECK:                 %[[BITCAST_11:.*]] = vector.bitcast %[[VAL_40]] : vector<16xi32> to vector<16xf32>
// CHECK:                 %[[SHUFFLE_52:.*]] = vector.shuffle %[[BITCAST_10]], %[[BITCAST_11]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[SHUFFLE_53:.*]] = vector.shuffle %[[BITCAST_10]], %[[BITCAST_11]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[SHUFFLE_54:.*]] = vector.shuffle %[[SHUFFLE_52]], %[[SHUFFLE_52]] [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[SHUFFLE_55:.*]] = vector.shuffle %[[SHUFFLE_53]], %[[SHUFFLE_53]] [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15] : vector<16xf32>, vector<16xf32>
// CHECK:                 %[[LOAD_45:.*]] = vector.load %[[SUBVIEW_4]]{{\[}}%[[VAL_37]], %[[CONSTANT_17]]] : memref<16x16xf32, strided<[128, 1], offset: ?>>, vector<16xf32>
// CHECK:                 %[[ADDF_2:.*]] = arith.addf %[[SHUFFLE_54]], %[[LOAD_45]] : vector<16xf32>
// CHECK:                 vector.store %[[ADDF_2]], %[[ALLOCA_2]]{{\[}}%[[VAL_37]], %[[CONSTANT_17]]] : memref<16x16xf32>, vector<16xf32>
// CHECK:                 %[[LOAD_46:.*]] = vector.load %[[SUBVIEW_5]]{{\[}}%[[VAL_37]], %[[CONSTANT_17]]] : memref<16x16xf32, strided<[128, 1], offset: ?>>, vector<16xf32>
// CHECK:                 %[[ADDF_3:.*]] = arith.addf %[[SHUFFLE_55]], %[[LOAD_46]] : vector<16xf32>
// CHECK:                 vector.store %[[ADDF_3]], %[[ALLOCA_3]]{{\[}}%[[VAL_37]], %[[CONSTANT_17]]] : memref<16x16xf32>, vector<16xf32>
// CHECK:               }
// CHECK:               %[[TRANSFER_READ_2:.*]] = vector.transfer_read %[[ALLOCA_2]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_17]]], %[[POISON_0]] {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
// CHECK:               %[[TRANSFER_READ_3:.*]] = vector.transfer_read %[[ALLOCA_3]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_17]]], %[[POISON_0]] {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>
// CHECK:               vector.transfer_write %[[TRANSFER_READ_3]], %[[SUBVIEW_0]]{{\[}}%[[CONSTANT_16]], %[[CONSTANT_16]]] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
// CHECK:               vector.transfer_write %[[TRANSFER_READ_2]], %[[SUBVIEW_0]]{{\[}}%[[CONSTANT_16]], %[[CONSTANT_17]]] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
// CHECK:               vector.transfer_write %[[TRANSFER_READ_1]], %[[SUBVIEW_0]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_16]]] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
// CHECK:               vector.transfer_write %[[TRANSFER_READ_0]], %[[SUBVIEW_0]]{{\[}}%[[CONSTANT_17]], %[[CONSTANT_17]]] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[128, 1], offset: ?>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK-NOT:     vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xf8E4M3FN>
!memrefB = memref<32x128x128xf8E4M3FN>
!memrefC = memref<128x128xf32>
!subA = memref<1x32x16xf8E4M3FN, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x16x64xf8E4M3FN, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x64xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x16xf8E4M3FN>
!vecB = vector<1x16x16xf8E4M3FN>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_hf8_case3(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c48 = arith.constant 48 : index
  %0 = ub.poison : f32
  %1 = ub.poison : f8E4M3FN
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 64] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c0, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c0, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %7 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %8 = vector.transfer_read %subview[%c16, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %9 = vector.transfer_read %subview[%c16, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %10:8 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5, %arg10 = %6,
                    %arg11 = %7, %arg12 = %8, %arg13 = %9)
          -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
        %11:8 = scf.for %arg14 = %c0 to %c128 step %c16
            iter_args(%arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8,
                      %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11,
                      %arg21 = %arg12, %arg22 = %arg13)
            -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg14]
              [1, 32, 16] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg14, %arg4]
              [1, 16, 64] [1, 1, 1] : !memrefB to !subB
          %12 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %13 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %14 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %15 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %16 = vector.transfer_read %subview_1[%c0, %c0, %c32], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %17 = vector.transfer_read %subview_1[%c0, %c0, %c48], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %18 = vector.contract #trait %12, %14, %arg15
: !vecA, !vecB into !vecC
          %19 = vector.contract #trait %12, %15, %arg16
: !vecA, !vecB into !vecC
          %20 = vector.contract #trait %12, %16, %arg17
: !vecA, !vecB into !vecC
          %21 = vector.contract #trait %12, %17, %arg18
: !vecA, !vecB into !vecC
          %22 = vector.contract #trait %13, %14, %arg19
: !vecA, !vecB into !vecC
          %23 = vector.contract #trait %13, %15, %arg20
: !vecA, !vecB into !vecC
          %24 = vector.contract #trait %13, %16, %arg21
: !vecA, !vecB into !vecC
          %25 = vector.contract #trait %13, %17, %arg22
: !vecA, !vecB into !vecC
          scf.yield %18, %19, %20, %21, %22, %23, %24, %25
              : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %11#0, %11#1, %11#2, %11#3, %11#4, %11#5, %11#6, %11#7
            : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %10#7, %subview[%c16, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#6, %subview[%c16, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#5, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#4, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#3, %subview[%c0, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#2, %subview[%c0, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_hf8_case3
// CHECK:           x86.ace.bsr_init
// CHECK-COUNT-8:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-32:  x86.ace.tile_mulf {{.*}} : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-8:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_hf8_case3
// SHUFFLE-COUNT-88: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xi8>
!memrefB = memref<32x128x128xi8>
!memrefC = memref<128x128xi32>
!subA = memref<1x32x16xi8, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x16x32xi8, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xi32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x16xi8>
!vecB = vector<1x16x16xi8>
!vecC = vector<16x16xi32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_i8_case3(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c16
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10]
              [1, 32, 16] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4]
              [1, 16, 32] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_i8_case3
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-COUNT-16:  x86.ace.tile_muli {{.*}} : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
// CHECK-NOT:       x86.ace.tile_muli
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_i8_case3
// SHUFFLE-COUNT-64: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xbf16>
!memrefB = memref<32x128x128xbf16>
!memrefC = memref<128x128xf32>
!subA = memref<1x2x32xbf16, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x2x64xbf16, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x64xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x2x16xbf16>
!vecB = vector<1x2x16xbf16>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_bf16_case4(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c48 = arith.constant 48 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 64] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c0, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c0, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %7 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %8 = vector.transfer_read %subview[%c16, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %9 = vector.transfer_read %subview[%c16, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %10:8 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5, %arg10 = %6,
                    %arg11 = %7, %arg12 = %8, %arg13 = %9)
          -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
        %11:8 = scf.for %arg14 = %c0 to %c128 step %c2
            iter_args(%arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8,
                      %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11,
                      %arg21 = %arg12, %arg22 = %arg13)
            -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg14, %arg3]
              [1, 2, 32] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg14, %arg4]
              [1, 2, 64] [1, 1, 1] : !memrefB to !subB
          %12 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %13 = vector.transfer_read %subview_0[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %14 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %15 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %16 = vector.transfer_read %subview_1[%c0, %c0, %c32], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %17 = vector.transfer_read %subview_1[%c0, %c0, %c48], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %18 = vector.contract #trait %12, %14, %arg15
: !vecA, !vecB into !vecC
          %19 = vector.contract #trait %12, %15, %arg16
: !vecA, !vecB into !vecC
          %20 = vector.contract #trait %12, %16, %arg17
: !vecA, !vecB into !vecC
          %21 = vector.contract #trait %12, %17, %arg18
: !vecA, !vecB into !vecC
          %22 = vector.contract #trait %13, %14, %arg19
: !vecA, !vecB into !vecC
          %23 = vector.contract #trait %13, %15, %arg20
: !vecA, !vecB into !vecC
          %24 = vector.contract #trait %13, %16, %arg21
: !vecA, !vecB into !vecC
          %25 = vector.contract #trait %13, %17, %arg22
: !vecA, !vecB into !vecC
          scf.yield %18, %19, %20, %21, %22, %23, %24, %25
              : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %11#0, %11#1, %11#2, %11#3, %11#4, %11#5, %11#6, %11#7
            : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %10#7, %subview[%c16, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#6, %subview[%c16, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#5, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#4, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#3, %subview[%c0, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#2, %subview[%c0, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_bf16_case4
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-8:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-8:   x86.ace.tile_mulf {{.*}} : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-8:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_bf16_case4
// SHUFFLE-COUNT-22: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xf8E4M3FN>
!memrefB = memref<32x128x128xf8E4M3FN>
!memrefC = memref<128x128xf32>
!subA = memref<1x4x32xf8E4M3FN, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x4x64xf8E4M3FN, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x64xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x4x16xf8E4M3FN>
!vecB = vector<1x4x16xf8E4M3FN>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_hf8_case4(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c48 = arith.constant 48 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : f8E4M3FN
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 64] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c0, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c0, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %7 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %8 = vector.transfer_read %subview[%c16, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %9 = vector.transfer_read %subview[%c16, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %10:8 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5, %arg10 = %6,
                    %arg11 = %7, %arg12 = %8, %arg13 = %9)
          -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
        %11:8 = scf.for %arg14 = %c0 to %c128 step %c4
            iter_args(%arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8,
                      %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11,
                      %arg21 = %arg12, %arg22 = %arg13)
            -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg14, %arg3]
              [1, 4, 32] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg14, %arg4]
              [1, 4, 64] [1, 1, 1] : !memrefB to !subB
          %12 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %13 = vector.transfer_read %subview_0[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %14 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %15 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %16 = vector.transfer_read %subview_1[%c0, %c0, %c32], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %17 = vector.transfer_read %subview_1[%c0, %c0, %c48], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %18 = vector.contract #trait %12, %14, %arg15
: !vecA, !vecB into !vecC
          %19 = vector.contract #trait %12, %15, %arg16
: !vecA, !vecB into !vecC
          %20 = vector.contract #trait %12, %16, %arg17
: !vecA, !vecB into !vecC
          %21 = vector.contract #trait %12, %17, %arg18
: !vecA, !vecB into !vecC
          %22 = vector.contract #trait %13, %14, %arg19
: !vecA, !vecB into !vecC
          %23 = vector.contract #trait %13, %15, %arg20
: !vecA, !vecB into !vecC
          %24 = vector.contract #trait %13, %16, %arg21
: !vecA, !vecB into !vecC
          %25 = vector.contract #trait %13, %17, %arg22
: !vecA, !vecB into !vecC
          scf.yield %18, %19, %20, %21, %22, %23, %24, %25
              : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %11#0, %11#1, %11#2, %11#3, %11#4, %11#5, %11#6, %11#7
            : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %10#7, %subview[%c16, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#6, %subview[%c16, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#5, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#4, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#3, %subview[%c0, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#2, %subview[%c0, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_hf8_case4
// CHECK:           x86.ace.bsr_init
// CHECK-COUNT-8:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-8:   x86.ace.tile_mulf {{.*}} : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-8:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_hf8_case4
// SHUFFLE-COUNT-30: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xi8>
!memrefB = memref<32x128x128xi8>
!memrefC = memref<128x128xi32>
!subA = memref<1x4x32xi8, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x4x32xi8, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xi32, strided<[128, 1], offset: ?>>
!vecA = vector<1x4x16xi8>
!vecB = vector<1x4x16xi8>
!vecC = vector<16x16xi32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_i8_case4(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c4
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg10, %arg3]
              [1, 4, 32] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4]
              [1, 4, 32] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_i8_case4
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-COUNT-4:   x86.ace.tile_muli {{.*}} : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
// CHECK-NOT:       x86.ace.tile_muli
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_i8_case4
// SHUFFLE-COUNT-12: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xbf16>
!memrefB = memref<32x128x128xbf16>
!memrefC = memref<128x128xf32>
!subA = memref<1x32x8xbf16, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x32x8xbf16, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x8xbf16>
!vecB = vector<1x16x8xbf16>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_bf16_case5(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c8
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10]
              [1, 32, 8] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg4, %arg10]
              [1, 32, 8] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_bf16_case5
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-16:  x86.ace.tile_mulf {{.*}} : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_bf16_case5
// SHUFFLE-COUNT-80: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xf8E5M2>
!memrefB = memref<32x128x128xf8E5M2>
!memrefC = memref<128x128xf32>
!subA = memref<1x32x16xf8E5M2, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x32x16xf8E5M2, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x16xf8E5M2>
!vecB = vector<1x16x16xf8E5M2>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_bf8_case5(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %0 = ub.poison : f32
  %1 = ub.poison : f8E5M2
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c16
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10]
              [1, 32, 16] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg4, %arg10]
              [1, 32, 16] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_bf8_case5
// CHECK:           x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-16:  x86.ace.tile_mulf {{.*}} : vector<64xf8E5M2>, vector<64xf8E5M2>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_bf8_case5
// SHUFFLE-COUNT-80: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xbf16>
!memrefB = memref<32x128x128xbf16>
!memrefC = memref<128x128xf32>
!subA = memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x32x8xbf16, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x8x16xbf16>
!vecB = vector<1x16x8xbf16>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_bf16_case6(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c8
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg10, %arg3]
              [1, 8, 32] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg4, %arg10]
              [1, 32, 8] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_bf16_case6
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-16:  x86.ace.tile_mulf {{.*}} : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_bf16_case6
// SHUFFLE-COUNT-48: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xi8>
!memrefB = memref<32x128x128xi8>
!memrefC = memref<128x128xi32>
!subA = memref<1x16x32xi8, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x32x16xi8, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xi32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x16xi8>
!vecB = vector<1x16x16xi8>
!vecC = vector<16x16xi32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_i8_case6(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c16
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg10, %arg3]
              [1, 16, 32] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg4, %arg10]
              [1, 32, 16] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_i8_case6
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-COUNT-16:  x86.ace.tile_muli {{.*}} : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
// CHECK-NOT:       x86.ace.tile_muli
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_i8_case6
// SHUFFLE-COUNT-64: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x64x128x2xbf16>
!memrefB = memref<32x64x128x2xbf16>
!memrefC = memref<128x128xf32>
!subA = memref<1x1x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>
!subB = memref<1x1x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>
!subC = memref<32x32xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x1x16x2xbf16>
!vecB = vector<1x1x16x2xbf16>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction",
                    "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_bf16_case1(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c64 step %c1
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg10, %arg3, 0]
              [1, 1, 32, 2] [1, 1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4, 0]
              [1, 1, 32, 2] [1, 1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_bf16_case1
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-4:   x86.ace.tile_mulf {{.*}} : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_bf16_case1
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x32x128x4xf8E4M3FN>
!memrefB = memref<32x32x128x4xf8E4M3FN>
!memrefC = memref<128x128xf32>
!subA = memref<1x1x32x4xf8E4M3FN, strided<[16384, 512, 4, 1], offset: ?>>
!subB = memref<1x1x64x4xf8E4M3FN, strided<[16384, 512, 4, 1], offset: ?>>
!subC = memref<32x64xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x1x16x4xf8E4M3FN>
!vecB = vector<1x1x16x4xf8E4M3FN>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction",
                    "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_hf8_case1(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c48 = arith.constant 48 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : f8E4M3FN
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c64 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 64] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c0, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c0, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %7 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %8 = vector.transfer_read %subview[%c16, %c32], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %9 = vector.transfer_read %subview[%c16, %c48], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %10:8 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5, %arg10 = %6,
                    %arg11 = %7, %arg12 = %8, %arg13 = %9)
          -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
        %11:8 = scf.for %arg14 = %c0 to %c32 step %c1
            iter_args(%arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8,
                      %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11,
                      %arg21 = %arg12, %arg22 = %arg13)
            -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg14, %arg3, 0]
              [1, 1, 32, 4] [1, 1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg14, %arg4, 0]
              [1, 1, 64, 4] [1, 1, 1, 1] : !memrefB to !subB
          %12 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %13 = vector.transfer_read %subview_0[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %14 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %15 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %16 = vector.transfer_read %subview_1[%c0, %c0, %c32, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %17 = vector.transfer_read %subview_1[%c0, %c0, %c48, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %18 = vector.contract #trait %12, %14, %arg15
: !vecA, !vecB into !vecC
          %19 = vector.contract #trait %12, %15, %arg16
: !vecA, !vecB into !vecC
          %20 = vector.contract #trait %12, %16, %arg17
: !vecA, !vecB into !vecC
          %21 = vector.contract #trait %12, %17, %arg18
: !vecA, !vecB into !vecC
          %22 = vector.contract #trait %13, %14, %arg19
: !vecA, !vecB into !vecC
          %23 = vector.contract #trait %13, %15, %arg20
: !vecA, !vecB into !vecC
          %24 = vector.contract #trait %13, %16, %arg21
: !vecA, !vecB into !vecC
          %25 = vector.contract #trait %13, %17, %arg22
: !vecA, !vecB into !vecC
          scf.yield %18, %19, %20, %21, %22, %23, %24, %25
              : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %11#0, %11#1, %11#2, %11#3, %11#4, %11#5, %11#6, %11#7
            : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %10#7, %subview[%c16, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#6, %subview[%c16, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#5, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#4, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#3, %subview[%c0, %c48]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#2, %subview[%c0, %c32]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %10#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_hf8_case1
// CHECK:           x86.ace.bsr_init
// CHECK-COUNT-8:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-8:   x86.ace.tile_mulf {{.*}} : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-8:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_hf8_case1
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x32x128x4xi8>
!memrefB = memref<32x32x128x4xi8>
!memrefC = memref<128x128xi32>
!subA = memref<1x1x32x4xi8, strided<[16384, 512, 4, 1], offset: ?>>
!subB = memref<1x1x32x4xi8, strided<[16384, 512, 4, 1], offset: ?>>
!subC = memref<32x32xi32, strided<[128, 1], offset: ?>>
!vecA = vector<1x1x16x4xi8>
!vecB = vector<1x1x16x4xi8>
!vecC = vector<16x16xi32>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction",
                    "reduction"],
  kind = #vector.kind<add>
}

func.func @brgemm_i8_case1(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c32 step %c1
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg10, %arg3, 0]
              [1, 1, 32, 4] [1, 1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4, 0]
              [1, 1, 32, 4] [1, 1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @brgemm_i8_case1
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-COUNT-4:   x86.ace.tile_muli {{.*}} : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
// CHECK-NOT:       x86.ace.tile_muli
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @brgemm_i8_case1
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x64x128x2xbf16>
!memrefB = memref<32x64x128x2xbf16>
!memrefC = memref<32x128x128xf32>
!subA = memref<1x1x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>
!subB = memref<1x1x32x2xbf16, strided<[16384, 256, 2, 1], offset: ?>>
!subC = memref<1x32x32xf32, strided<[16384, 128, 1], offset: ?>>
!vecA = vector<1x1x16x2xbf16>
!vecB = vector<1x1x16x2xbf16>
!vecC = vector<1x16x16xf32>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["parallel", "parallel", "parallel", "reduction",
                    "reduction"],
  kind = #vector.kind<add>
}

func.func @batch_matmul_bf16_case1(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %subview = memref.subview %arg2[%arg5, %arg3, %arg4]
            [1, 32, 32] [1, 1, 1] : !memrefC to !subC
        %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %3 = vector.transfer_read %subview[%c0, %c0, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %4 = vector.transfer_read %subview[%c0, %c16, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %5 = vector.transfer_read %subview[%c0, %c16, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %6:4 = scf.for %arg6 = %c0 to %c64 step %c1
            iter_args(%arg7 = %2, %arg8 = %3, %arg9 = %4, %arg10 = %5)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg6, %arg3, 0]
              [1, 1, 32, 2] [1, 1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4, 0]
              [1, 1, 32, 2] [1, 1, 1, 1] : !memrefB to !subB
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c16, %c0], %1
              {in_bounds = [true, true, true, true]} : !subB, !vecB
          %11 = vector.contract #trait %7, %9, %arg7 : !vecA, !vecB into !vecC
          %12 = vector.contract #trait %7, %10, %arg8 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %9, %arg9 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %8, %10, %arg10 : !vecA, !vecB into !vecC
          scf.yield %11, %12, %13, %14 : !vecC, !vecC, !vecC, !vecC
        }
        vector.transfer_write %6#3, %subview[%c0, %c16, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#2, %subview[%c0, %c16, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#1, %subview[%c0, %c0, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#0, %subview[%c0, %c0, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
      }
    }
  }
  return
}

// CHECK-LABEL:     func.func @batch_matmul_bf16_case1
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-4:   x86.ace.tile_mulf {{.*}} : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @batch_matmul_bf16_case1
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<64x128x2xbf16>
!memrefB = memref<64x128x2xbf16>
!memrefC = memref<128x128xf32>
!subA = memref<1x32x2xbf16, strided<[256, 2, 1], offset: ?>>
!subB = memref<1x32x2xbf16, strided<[256, 2, 1], offset: ?>>
!subC = memref<32x32xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x2xbf16>
!vecB = vector<1x16x2xbf16>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"],
  kind = #vector.kind<add>
}

func.func @matmul_bf16_case1(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c64 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %subview_0 = memref.subview %arg0[%arg5, %arg3, 0] [1, 32, 2] [1, 1, 1]
            : !memrefA to !subA
        %subview_1 = memref.subview %arg1[%arg5, %arg4, 0] [1, 32, 2] [1, 1, 1]
            : !memrefB to !subB
        %7 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
            {in_bounds = [true, true, true]} : !subA, !vecA
        %8 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
            {in_bounds = [true, true, true]} : !subA, !vecA
        %9 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
            {in_bounds = [true, true, true]} : !subB, !vecB
        %10 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1
            {in_bounds = [true, true, true]} : !subB, !vecB
        %11 = vector.contract #trait %7, %9, %arg6 : !vecA, !vecB into !vecC
        %12 = vector.contract #trait %7, %10, %arg7 : !vecA, !vecB into !vecC
        %13 = vector.contract #trait %8, %9, %arg8 : !vecA, !vecB into !vecC
        %14 = vector.contract #trait %8, %10, %arg9 : !vecA, !vecB into !vecC
        scf.yield %11, %12, %13, %14 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @matmul_bf16_case1
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-4:   x86.ace.tile_mulf {{.*}} : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @matmul_bf16_case1
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xf8E5M2>
!memrefB = memref<32x128x128xf8E5M2>
!memrefC = memref<32x128x128xf32>
!subA = memref<1x32x16xf8E5M2, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x16x64xf8E5M2, strided<[16384, 128, 1], offset: ?>>
!subC = memref<1x32x64xf32, strided<[16384, 128, 1], offset: ?>>
!vecA = vector<1x16x16xf8E5M2>
!vecB = vector<1x16x16xf8E5M2>
!vecC = vector<1x16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["parallel", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @batch_matmul_bf8_case3(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c48 = arith.constant 48 : index
  %0 = ub.poison : f32
  %1 = ub.poison : f8E5M2
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c64 {
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %subview = memref.subview %arg2[%arg5, %arg3, %arg4]
            [1, 32, 64] [1, 1, 1] : !memrefC to !subC
        %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %3 = vector.transfer_read %subview[%c0, %c0, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %4 = vector.transfer_read %subview[%c0, %c0, %c32], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %5 = vector.transfer_read %subview[%c0, %c0, %c48], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %6 = vector.transfer_read %subview[%c0, %c16, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %7 = vector.transfer_read %subview[%c0, %c16, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %8 = vector.transfer_read %subview[%c0, %c16, %c32], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %9 = vector.transfer_read %subview[%c0, %c16, %c48], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %10:8 = scf.for %arg6 = %c0 to %c128 step %c16
            iter_args(%arg7 = %2, %arg8 = %3, %arg9 = %4, %arg10 = %5,
                      %arg11 = %6, %arg12 = %7, %arg13 = %8, %arg14 = %9)
            -> (!vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6]
              [1, 32, 16] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4]
              [1, 16, 64] [1, 1, 1] : !memrefB to !subB
          %11 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %12 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %13 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %14 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %15 = vector.transfer_read %subview_1[%c0, %c0, %c32], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %16 = vector.transfer_read %subview_1[%c0, %c0, %c48], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %17 = vector.contract #trait %11, %13, %arg7 : !vecA, !vecB into !vecC
          %18 = vector.contract #trait %11, %14, %arg8 : !vecA, !vecB into !vecC
          %19 = vector.contract #trait %11, %15, %arg9 : !vecA, !vecB into !vecC
          %20 = vector.contract #trait %11, %16, %arg10
: !vecA, !vecB into !vecC
          %21 = vector.contract #trait %12, %13, %arg11
: !vecA, !vecB into !vecC
          %22 = vector.contract #trait %12, %14, %arg12
: !vecA, !vecB into !vecC
          %23 = vector.contract #trait %12, %15, %arg13
: !vecA, !vecB into !vecC
          %24 = vector.contract #trait %12, %16, %arg14
: !vecA, !vecB into !vecC
          scf.yield %17, %18, %19, %20, %21, %22, %23, %24
              : !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC, !vecC
        }
        vector.transfer_write %10#7, %subview[%c0, %c16, %c48]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %10#6, %subview[%c0, %c16, %c32]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %10#5, %subview[%c0, %c16, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %10#4, %subview[%c0, %c16, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %10#3, %subview[%c0, %c0, %c48]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %10#2, %subview[%c0, %c0, %c32]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %10#1, %subview[%c0, %c0, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %10#0, %subview[%c0, %c0, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
      }
    }
  }
  return
}

// CHECK-LABEL:     func.func @batch_matmul_bf8_case3
// CHECK:           x86.ace.bsr_init
// CHECK-COUNT-8:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-32:  x86.ace.tile_mulf {{.*}} : vector<64xf8E5M2>, vector<64xf8E5M2>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-8:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @batch_matmul_bf8_case3
// SHUFFLE-COUNT-88: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xi8>
!memrefB = memref<32x128x128xi8>
!memrefC = memref<32x128x128xi32>
!subA = memref<1x4x32xi8, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x4x32xi8, strided<[16384, 128, 1], offset: ?>>
!subC = memref<1x32x32xi32, strided<[16384, 128, 1], offset: ?>>
!vecA = vector<1x4x16xi8>
!vecB = vector<1x4x16xi8>
!vecC = vector<1x16x16xi32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["parallel", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @batch_matmul_i8_case4(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %subview = memref.subview %arg2[%arg5, %arg3, %arg4]
            [1, 32, 32] [1, 1, 1] : !memrefC to !subC
        %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %3 = vector.transfer_read %subview[%c0, %c0, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %4 = vector.transfer_read %subview[%c0, %c16, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %5 = vector.transfer_read %subview[%c0, %c16, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %6:4 = scf.for %arg6 = %c0 to %c128 step %c4
            iter_args(%arg7 = %2, %arg8 = %3, %arg9 = %4, %arg10 = %5)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg6, %arg3]
              [1, 4, 32] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4]
              [1, 4, 32] [1, 1, 1] : !memrefB to !subB
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.contract #trait %7, %9, %arg7 : !vecA, !vecB into !vecC
          %12 = vector.contract #trait %7, %10, %arg8 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %9, %arg9 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %8, %10, %arg10 : !vecA, !vecB into !vecC
          scf.yield %11, %12, %13, %14 : !vecC, !vecC, !vecC, !vecC
        }
        vector.transfer_write %6#3, %subview[%c0, %c16, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#2, %subview[%c0, %c16, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#1, %subview[%c0, %c0, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#0, %subview[%c0, %c0, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
      }
    }
  }
  return
}

// CHECK-LABEL:     func.func @batch_matmul_i8_case4
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
// CHECK-COUNT-4:   x86.ace.tile_muli {{.*}} : vector<64xi8>, vector<64xi8>, !x86.amx.tile<16x16xi32>
// CHECK-NOT:       x86.ace.tile_muli
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @batch_matmul_i8_case4
// SHUFFLE-COUNT-12: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xf8E5M2>
!memrefB = memref<32x128x128xf8E5M2>
!memrefC = memref<32x128x128xf32>
!subA = memref<1x32x16xf8E5M2, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x32x16xf8E5M2, strided<[16384, 128, 1], offset: ?>>
!subC = memref<1x32x32xf32, strided<[16384, 128, 1], offset: ?>>
!vecA = vector<1x16x16xf8E5M2>
!vecB = vector<1x16x16xf8E5M2>
!vecC = vector<1x16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["parallel", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @batch_matmul_bf8_case5(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %0 = ub.poison : f32
  %1 = ub.poison : f8E5M2
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %subview = memref.subview %arg2[%arg5, %arg3, %arg4]
            [1, 32, 32] [1, 1, 1] : !memrefC to !subC
        %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %3 = vector.transfer_read %subview[%c0, %c0, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %4 = vector.transfer_read %subview[%c0, %c16, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %5 = vector.transfer_read %subview[%c0, %c16, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %6:4 = scf.for %arg6 = %c0 to %c128 step %c16
            iter_args(%arg7 = %2, %arg8 = %3, %arg9 = %4, %arg10 = %5)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6]
              [1, 32, 16] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg4, %arg6]
              [1, 32, 16] [1, 1, 1] : !memrefB to !subB
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %8 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %10 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.contract #trait %7, %9, %arg7 : !vecA, !vecB into !vecC
          %12 = vector.contract #trait %7, %10, %arg8 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %9, %arg9 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %8, %10, %arg10 : !vecA, !vecB into !vecC
          scf.yield %11, %12, %13, %14 : !vecC, !vecC, !vecC, !vecC
        }
        vector.transfer_write %6#3, %subview[%c0, %c16, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#2, %subview[%c0, %c16, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#1, %subview[%c0, %c0, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#0, %subview[%c0, %c0, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
      }
    }
  }
  return
}

// CHECK-LABEL:     func.func @batch_matmul_bf8_case5
// CHECK:           x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-16:  x86.ace.tile_mulf {{.*}} : vector<64xf8E5M2>, vector<64xf8E5M2>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @batch_matmul_bf8_case5
// SHUFFLE-COUNT-80: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!memrefA = memref<32x128x128xbf16>
!memrefB = memref<32x128x128xbf16>
!memrefC = memref<32x128x128xf32>
!subA = memref<1x8x32xbf16, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x32x8xbf16, strided<[16384, 128, 1], offset: ?>>
!subC = memref<1x32x32xf32, strided<[16384, 128, 1], offset: ?>>
!vecA = vector<1x8x16xbf16>
!vecB = vector<1x16x8xbf16>
!vecC = vector<1x16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["parallel", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @batch_matmul_bf16_case6(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %subview = memref.subview %arg2[%arg5, %arg3, %arg4]
            [1, 32, 32] [1, 1, 1] : !memrefC to !subC
        %2 = vector.transfer_read %subview[%c0, %c0, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %3 = vector.transfer_read %subview[%c0, %c0, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %4 = vector.transfer_read %subview[%c0, %c16, %c0], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %5 = vector.transfer_read %subview[%c0, %c16, %c16], %0
            {in_bounds = [true, true, true]} : !subC, !vecC
        %6:4 = scf.for %arg6 = %c0 to %c128 step %c8
            iter_args(%arg7 = %2, %arg8 = %3, %arg9 = %4, %arg10 = %5)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg6, %arg3]
              [1, 8, 32] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg4, %arg6]
              [1, 32, 8] [1, 1, 1] : !memrefB to !subB
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %10 = vector.transfer_read %subview_1[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.contract #trait %7, %9, %arg7 : !vecA, !vecB into !vecC
          %12 = vector.contract #trait %7, %10, %arg8 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %9, %arg9 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %8, %10, %arg10 : !vecA, !vecB into !vecC
          scf.yield %11, %12, %13, %14 : !vecC, !vecC, !vecC, !vecC
        }
        vector.transfer_write %6#3, %subview[%c0, %c16, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#2, %subview[%c0, %c16, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#1, %subview[%c0, %c0, %c16]
            {in_bounds = [true, true, true]} : !vecC, !subC
        vector.transfer_write %6#0, %subview[%c0, %c0, %c0]
            {in_bounds = [true, true, true]} : !vecC, !subC
      }
    }
  }
  return
}

// CHECK-LABEL:     func.func @batch_matmul_bf16_case6
// CHECK-NOT:       x86.ace.bsr_init
// CHECK-COUNT-4:   x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
// CHECK-COUNT-16:  x86.ace.tile_mulf {{.*}} : vector<32xbf16>, vector<32xbf16>, !x86.amx.tile<16x16xf32>
// CHECK-NOT:       x86.ace.tile_mulf
// CHECK-COUNT-4:   x86.amx.tilemovrow
// CHECK-NOT:       x86.amx.tilemovrow
// CHECK-NOT:       vector.contract

// SHUFFLE-LABEL:    func.func @batch_matmul_bf16_case6
// SHUFFLE-COUNT-48: vector.shuffle
// SHUFFLE-NOT:      vector.shuffle

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

// Case 3 bf16 expects K=8 per contraction; K=16 must not be lowered.

!memrefA = memref<32x128x128xbf16>
!memrefB = memref<32x128x128xbf16>
!memrefC = memref<128x128xf32>
!subA = memref<1x32x16xbf16, strided<[16384, 128, 1], offset: ?>>
!subB = memref<1x16x32xbf16, strided<[16384, 128, 1], offset: ?>>
!subC = memref<32x32xf32, strided<[128, 1], offset: ?>>
!vecA = vector<1x16x16xbf16>
!vecB = vector<1x16x16xbf16>
!vecC = vector<16x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#trait = {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

func.func @negative_brgemm_bf16_case3_wrong_k(%arg0: !memrefA, %arg1: !memrefB,
    %arg2: !memrefC) {
  %c16 = arith.constant 16 : index
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c128 step %c32 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
          : !memrefC to !subC
      %2 = vector.transfer_read %subview[%c0, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %4 = vector.transfer_read %subview[%c16, %c0], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %5 = vector.transfer_read %subview[%c16, %c16], %0
          {in_bounds = [true, true]} : !subC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c32 step %c1
          iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5)
          -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c128 step %c16
            iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8,
                      %arg14 = %arg9)
            -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10]
              [1, 32, 16] [1, 1, 1] : !memrefA to !subA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4]
              [1, 16, 32] [1, 1, 1] : !memrefB to !subB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c16, %c0], %1
              {in_bounds = [true, true, true]} : !subA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1
              {in_bounds = [true, true, true]} : !subB, !vecB
          %12 = vector.contract #trait %8, %10, %arg11 : !vecA, !vecB into !vecC
          %13 = vector.contract #trait %8, %11, %arg12 : !vecA, !vecB into !vecC
          %14 = vector.contract #trait %9, %10, %arg13 : !vecA, !vecB into !vecC
          %15 = vector.contract #trait %9, %11, %arg14 : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c16, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#2, %subview[%c16, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#1, %subview[%c0, %c16]
          {in_bounds = [true, true]} : !vecC, !subC
      vector.transfer_write %6#0, %subview[%c0, %c0]
          {in_bounds = [true, true]} : !vecC, !subC
    }
  }
  return
}

// CHECK-LABEL:     func.func @negative_brgemm_bf16_case3_wrong_k
// CHECK-NOT:       x86.ace
// CHECK-COUNT-4:   vector.contract
// CHECK-NOT:       x86.ace

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_ace_outer_product
    } : !transform.any_op
    transform.yield
  }
}


