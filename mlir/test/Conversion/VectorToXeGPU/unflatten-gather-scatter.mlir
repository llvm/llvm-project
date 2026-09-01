// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

// Frontends often emit a gather/scatter with 1-D operands, obtained by
// shape_cast-ing N-D indices and masks. The conversion undoes that first, so
// the resulting xegpu.load / xegpu.store keeps the N-D shape of the accessed
// data, which is the shape the XeGPU layouts downstream are expressed in.

gpu.module @xevm_module {
gpu.func @gather_2d_from_flat(%src: memref<?xbf16, strided<[1], offset: ?>>,
    %idx: vector<128x64xindex>, %mask: vector<128x64xi1>) -> vector<128x64xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8192xbf16>
  %flat_idx = vector.shape_cast %idx : vector<128x64xindex> to vector<8192xindex>
  %flat_mask = vector.shape_cast %mask : vector<128x64xi1> to vector<8192xi1>
  %flat_res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %cst
    : memref<?xbf16, strided<[1], offset: ?>>, vector<8192xindex>, vector<8192xi1>,
      vector<8192xbf16> into vector<8192xbf16>
  %res = vector.shape_cast %flat_res : vector<8192xbf16> to vector<128x64xbf16>
  gpu.return %res : vector<128x64xbf16>
}

// CHECK-LABEL: @gather_2d_from_flat(
// CHECK-SAME:    %[[SRC:.+]]: memref<?xbf16, strided<[1], offset: ?>>,
// CHECK-SAME:    %[[IDX:.+]]: vector<128x64xindex>, %[[MASK:.+]]: vector<128x64xi1>
// The flat pass-thru constant is reshaped rather than shape_cast.
// CHECK:         %[[PASS_THRU:.+]] = arith.constant dense<0.000000e+00> : vector<128x64xbf16>
// CHECK-NOT:     vector.shape_cast
// CHECK:         %[[VEC:.+]] = xegpu.load %{{.+}}[%{{.+}}], %[[MASK]]
// CHECK-SAME:      : i64, vector<128x64xindex>, vector<128x64xi1> -> vector<128x64xbf16>
// CHECK:         %[[RES:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]]
// CHECK-NOT:     vector.shape_cast
// CHECK:         gpu.return %[[RES]] : vector<128x64xbf16>
}

// -----

gpu.module @xevm_module {
gpu.func @scatter_2d_from_flat(%src: memref<?xbf16, strided<[1], offset: ?>>,
    %idx: vector<128x128xindex>, %mask: vector<128x128xi1>,
    %val: vector<128x128xbf16>) {
  %c0 = arith.constant 0 : index
  %flat_idx = vector.shape_cast %idx : vector<128x128xindex> to vector<16384xindex>
  %flat_mask = vector.shape_cast %mask : vector<128x128xi1> to vector<16384xi1>
  %flat_val = vector.shape_cast %val : vector<128x128xbf16> to vector<16384xbf16>
  vector.scatter %src[%c0] [%flat_idx], %flat_mask, %flat_val
    : memref<?xbf16, strided<[1], offset: ?>>, vector<16384xindex>, vector<16384xi1>,
      vector<16384xbf16>
  gpu.return
}

// CHECK-LABEL: @scatter_2d_from_flat(
// CHECK-SAME:    %[[SRC:.+]]: memref<?xbf16, strided<[1], offset: ?>>,
// CHECK-SAME:    %[[IDX:.+]]: vector<128x128xindex>, %[[MASK:.+]]: vector<128x128xi1>,
// CHECK-SAME:    %[[VAL:.+]]: vector<128x128xbf16>
// CHECK-NOT:     vector.shape_cast
// CHECK:         xegpu.store %[[VAL]], %{{.+}}[%{{.+}}], %[[MASK]]
// CHECK-SAME:      : vector<128x128xbf16>, i64, vector<128x128xindex>, vector<128x128xi1>
}

// -----

// A splat mask / pass-thru is rebuilt at the N-D shape.
gpu.module @xevm_module {
gpu.func @gather_2d_splat_operands(%src: memref<?xf32>, %idx: vector<8x16xindex>,
    %p: i1) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
  %flat_idx = vector.shape_cast %idx : vector<8x16xindex> to vector<128xindex>
  %flat_mask = vector.broadcast %p : i1 to vector<128xi1>
  %flat_res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %cst
    : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
      into vector<128xf32>
  %res = vector.shape_cast %flat_res : vector<128xf32> to vector<8x16xf32>
  gpu.return %res : vector<8x16xf32>
}

// CHECK-LABEL: @gather_2d_splat_operands(
// CHECK-SAME:    %[[SRC:.+]]: memref<?xf32>, %[[IDX:.+]]: vector<8x16xindex>, %[[P:.+]]: i1
// CHECK-DAG:     %[[PASS_THRU:.+]] = arith.constant dense<1.000000e+00> : vector<8x16xf32>
// CHECK-DAG:     %[[MASK:.+]] = vector.broadcast %[[P]] : i1 to vector<8x16xi1>
// CHECK:         %[[VEC:.+]] = xegpu.load %{{.+}}[%[[IDX]]], %[[MASK]]
// CHECK-SAME:      : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// CHECK:         arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]]
}

// -----

// The mask cannot be un-flattened, so the gather is left flat: rewriting it
// would only move the shape_cast from the index to the mask operand.
gpu.module @xevm_module {
gpu.func @gather_opaque_mask_untouched(%src: memref<?xf32>, %idx: vector<8x16xindex>,
    %mask: vector<128xi1>, %pass_thru: vector<128xf32>) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %flat_idx = vector.shape_cast %idx : vector<8x16xindex> to vector<128xindex>
  %flat_res = vector.gather %src[%c0] [%flat_idx], %mask, %pass_thru
    : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
      into vector<128xf32>
  %res = vector.shape_cast %flat_res : vector<128xf32> to vector<8x16xf32>
  gpu.return %res : vector<8x16xf32>
}

// CHECK-LABEL: @gather_opaque_mask_untouched(
// CHECK-SAME:    %[[SRC:.+]]: memref<?xf32>, %[[IDX:.+]]: vector<8x16xindex>,
// CHECK-SAME:    %[[MASK:.+]]: vector<128xi1>, %[[PASS_THRU:.+]]: vector<128xf32>
// CHECK:         %[[FLAT_IDX:.+]] = vector.shape_cast %[[IDX]] : vector<8x16xindex> to vector<128xindex>
// CHECK:         %[[VEC:.+]] = xegpu.load %{{.+}}[%[[FLAT_IDX]]], %[[MASK]]
// CHECK-SAME:      : i64, vector<128xindex>, vector<128xi1> -> vector<128xf32>
// CHECK:         %[[SEL:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]]
// CHECK:         vector.shape_cast %[[SEL]] : vector<128xf32> to vector<8x16xf32>
}

// -----

// The pass-thru cannot be un-flattened, so the gather is left flat: rewriting
// it would only move the shape_cast from the index to the pass-thru operand.
gpu.module @xevm_module {
gpu.func @gather_opaque_pass_thru_untouched(%src: memref<?xf32>, %idx: vector<8x16xindex>,
    %mask: vector<8x16xi1>, %pass_thru: vector<128xf32>) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %flat_idx = vector.shape_cast %idx : vector<8x16xindex> to vector<128xindex>
  %flat_mask = vector.shape_cast %mask : vector<8x16xi1> to vector<128xi1>
  %res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %pass_thru
    : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
      into vector<128xf32>
  gpu.return %res : vector<128xf32>
}

// CHECK-LABEL: @gather_opaque_pass_thru_untouched(
// CHECK-SAME:    %[[SRC:.+]]: memref<?xf32>, %[[IDX:.+]]: vector<8x16xindex>,
// CHECK-SAME:    %[[MASK:.+]]: vector<8x16xi1>, %[[PASS_THRU:.+]]: vector<128xf32>
// CHECK:         %[[FLAT_IDX:.+]] = vector.shape_cast %[[IDX]] : vector<8x16xindex> to vector<128xindex>
// CHECK:         %[[FLAT_MASK:.+]] = vector.shape_cast %[[MASK]] : vector<8x16xi1> to vector<128xi1>
// CHECK:         %[[VEC:.+]] = xegpu.load %{{.+}}[%[[FLAT_IDX]]], %[[FLAT_MASK]]
// CHECK-SAME:      : i64, vector<128xindex>, vector<128xi1> -> vector<128xf32>
// CHECK:         %[[RES:.+]] = arith.select %[[FLAT_MASK]], %[[VEC]], %[[PASS_THRU]]
// CHECK:         gpu.return %[[RES]] : vector<128xf32>
}

// -----

// How the result is used does not matter: the rewrite always casts it back to
// the flat type. Here the use is flat, so that cast stays - but it replaces the
// two operand casts, so the access still ends up N-D and one op lighter.
gpu.module @xevm_module {
gpu.func @gather_flat_result_use(%src: memref<?xf32>, %idx: vector<8x16xindex>,
    %mask: vector<8x16xi1>) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
  %flat_idx = vector.shape_cast %idx : vector<8x16xindex> to vector<128xindex>
  %flat_mask = vector.shape_cast %mask : vector<8x16xi1> to vector<128xi1>
  %res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %cst
    : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
      into vector<128xf32>
  gpu.return %res : vector<128xf32>
}

// CHECK-LABEL: @gather_flat_result_use(
// CHECK-SAME:    %[[SRC:.+]]: memref<?xf32>, %[[IDX:.+]]: vector<8x16xindex>,
// CHECK-SAME:    %[[MASK:.+]]: vector<8x16xi1>
// CHECK:         %[[PASS_THRU:.+]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK:         %[[VEC:.+]] = xegpu.load %{{.+}}[%[[IDX]]], %[[MASK]]
// CHECK-SAME:      : i64, vector<8x16xindex>, vector<8x16xi1> -> vector<8x16xf32>
// CHECK:         %[[SEL:.+]] = arith.select %[[MASK]], %[[VEC]], %[[PASS_THRU]]
// CHECK:         %[[RES:.+]] = vector.shape_cast %[[SEL]] : vector<8x16xf32> to vector<128xf32>
// CHECK:         gpu.return %[[RES]] : vector<128xf32>
}
