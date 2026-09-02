// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8" --cse --verify-diagnostics --split-input-file %s | FileCheck %s
// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8" --verify-diagnostics --split-input-file %s | FileCheck %s --check-prefix=CHECK-NOCSE

// Tests for `ConvertExtractStridedMetadata` on sub-byte memrefs. The pattern
// rewrites `memref.extract_strided_metadata` on the original sub-byte source
// to an `extract_strided_metadata` on the i8 container produced by the
// type converter, scaling the runtime offset from i8 units back to
// emulated-element units and returning the original sub-byte sizes/strides
// as constants.

// -----

// Static zero offset, 1-D i4: emulated offset is constant 0 in i4 units; the
// size/stride constants match the original sub-byte shape/strides (8 / 1),
// not the i8 container shape/strides (4 / 1).

// CHECK-LABEL: func.func @extract_strided_metadata_i4
// CHECK-SAME:    %{{.+}}: memref<4xi8>
// CHECK-DAG:     %[[OFF:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[SZ:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[ST:.+]] = arith.constant 1 : index
// CHECK:         return %[[OFF]], %[[SZ]], %[[ST]]
func.func @extract_strided_metadata_i4(%arg: memref<8xi4>)
    -> (index, index, index) {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %arg :
    memref<8xi4> -> memref<i4>, index, index, index
  return %offset, %size, %stride : index, index, index
}

// -----

// Static non-zero offset, 1-D i4: srcStaticOffset on the original sub-byte
// type is already in emulated-element units, so it must be returned
// unchanged (8 here). The i8 container offset is 4 (8 i4-elements = 4
// i8-elements), but the pattern must NOT report 4 nor the over-scaled value
// 16 (= 8 * 2). Shape 6 keeps every constant (offset, size, stride)
// distinct so each can be matched independently.

// CHECK-LABEL: func.func @extract_strided_metadata_i4_static_offset
// CHECK-SAME:    %{{.+}}: memref<3xi8, strided<[1], offset: 4>>
// CHECK-DAG:     %[[OFF:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[SZ:.+]] = arith.constant 6 : index
// CHECK-DAG:     %[[ST:.+]] = arith.constant 1 : index
// CHECK:         return %[[OFF]], %[[SZ]], %[[ST]]
func.func @extract_strided_metadata_i4_static_offset(
    %arg: memref<6xi4, strided<[1], offset: 8>>) -> (index, index, index) {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %arg :
    memref<6xi4, strided<[1], offset: 8>> -> memref<i4>, index, index, index
  return %offset, %size, %stride : index, index, index
}

// -----

// Dynamic offset, 1-D i4: the runtime i8 offset returned by the inner
// `extract_strided_metadata` is multiplied by the scale factor
// `loadStoreBitwidth / elementBitwidth` (= 2 for i4 @ i8 container) to
// convert it back to i4-element units.

// CHECK-LABEL: func.func @extract_strided_metadata_i4_dynamic_offset
// CHECK-SAME:    %[[ARG:.+]]: memref<4xi8, strided<[1], offset: ?>>
// CHECK-DAG:     %[[SZ:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[ST:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[SCALE:.+]] = arith.constant 2 : index
// CHECK-DAG:     %{{.+}}, %[[I8OFF:.+]], %{{.+}}, %{{.+}} = memref.extract_strided_metadata %[[ARG]]
// CHECK-DAG:     %[[OFF:.+]] = arith.muli %[[I8OFF]], %[[SCALE]]
// CHECK:         return %[[OFF]], %[[SZ]], %[[ST]]
func.func @extract_strided_metadata_i4_dynamic_offset(
    %arg: memref<8xi4, strided<[1], offset: ?>>) -> (index, index, index) {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %arg :
    memref<8xi4, strided<[1], offset: ?>> -> memref<i4>, index, index, index
  return %offset, %size, %stride : index, index, index
}

// -----

// Dynamic offset, 1-D i2: scale factor is `loadStoreBitwidth /
// elementBitwidth` = 4. Validates that the scale is derived from the actual
// element width rather than hardcoded for i4.

// CHECK-LABEL: func.func @extract_strided_metadata_i2_dynamic_offset
// CHECK-SAME:    %[[ARG:.+]]: memref<4xi8, strided<[1], offset: ?>>
// CHECK-DAG:     %[[SZ:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[ST:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[SCALE:.+]] = arith.constant 4 : index
// CHECK-DAG:     %{{.+}}, %[[I8OFF:.+]], %{{.+}}, %{{.+}} = memref.extract_strided_metadata %[[ARG]]
// CHECK-DAG:     %[[OFF:.+]] = arith.muli %[[I8OFF]], %[[SCALE]]
// CHECK:         return %[[OFF]], %[[SZ]], %[[ST]]
func.func @extract_strided_metadata_i2_dynamic_offset(
    %arg: memref<16xi2, strided<[1], offset: ?>>) -> (index, index, index) {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %arg :
    memref<16xi2, strided<[1], offset: ?>> -> memref<i2>, index, index, index
  return %offset, %size, %stride : index, index, index
}

// -----

// Multi-dim sub-byte source: the converted source is linearized to 1-D, but
// the pattern still returns the original 2-D sizes/strides as constants
// derived from the original `MemRefType`. Shape <2x3> keeps strides distinct
// from sizes (strides [3, 1], sizes [2, 3]) so the per-result mapping is
// observable.

// CHECK-LABEL: func.func @extract_strided_metadata_i4_2d
// CHECK-SAME:    %{{.+}}: memref<3xi8>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK:         return %[[C0]], %[[C2]], %[[C3]], %[[C3]], %[[C1]]
func.func @extract_strided_metadata_i4_2d(%arg: memref<2x3xi4>)
    -> (index, index, index, index, index) {
  %base, %offset, %sz:2, %st:2 = memref.extract_strided_metadata %arg :
    memref<2x3xi4> -> memref<i4>, index, index, index, index, index
  return %offset, %sz#0, %sz#1, %st#0, %st#1 :
    index, index, index, index, index
}

// -----

// Base buffer of the rewrite is the i8 container's base, not the original
// sub-byte base. The no-CSE run line keeps the inner
// `extract_strided_metadata` so its `base_buffer` result is observable.

// CHECK-NOCSE-LABEL: func.func @extract_strided_metadata_i4_base_buffer
// CHECK-NOCSE-SAME:    %[[ARG:.+]]: memref<4xi8>
// CHECK-NOCSE:         %[[BASE:[A-Za-z0-9_]+]], %{{.+}}, %{{.+}}, %{{.+}} = memref.extract_strided_metadata %[[ARG]] : memref<4xi8> -> memref<i8>
// CHECK-NOCSE:         return %[[BASE]]
func.func @extract_strided_metadata_i4_base_buffer(%arg: memref<8xi4>)
    -> memref<i4> {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %arg :
    memref<8xi4> -> memref<i4>, index, index, index
  return %base : memref<i4>
}

// -----

// Rank-0 sub-byte source with zero offset: the converted source is a rank-0
// i8 memref, and the pattern returns only `{baseBuffer, emulatedOffset}` with
// no size/stride results (matching `extract_strided_metadata`'s rank-0
// signature). The emulated offset is the static zero constant.

// CHECK-LABEL: func.func @extract_strided_metadata_i4_rank0
// CHECK-SAME:    %{{.+}}: memref<i8>
// CHECK-DAG:     %[[OFF:.+]] = arith.constant 0 : index
// CHECK:         return %[[OFF]]
func.func @extract_strided_metadata_i4_rank0(%arg: memref<i4>) -> index {
  %base, %offset = memref.extract_strided_metadata %arg :
    memref<i4> -> memref<i4>, index
  return %offset : index
}

// -----

// Rank-0 sub-byte source with a dynamic offset: the converter produces a
// rank-0 i8 memref with a `strided<[], offset: ?>` layout (empty strides),
// and the pattern scales the runtime i8 offset back to i4 units via muli.

// CHECK-LABEL: func.func @extract_strided_metadata_i4_rank0_dynamic_offset
// CHECK-SAME:    %[[ARG:.+]]: memref<i8, strided<[], offset: ?>>
// CHECK-DAG:     %[[SCALE:.+]] = arith.constant 2 : index
// CHECK-DAG:     %{{.+}}, %[[I8OFF:.+]] = memref.extract_strided_metadata %[[ARG]]
// CHECK-DAG:     %[[OFF:.+]] = arith.muli %[[I8OFF]], %[[SCALE]]
// CHECK:         return %[[OFF]]
func.func @extract_strided_metadata_i4_rank0_dynamic_offset(
    %arg: memref<i4, strided<[], offset: ?>>) -> index {
  %base, %offset = memref.extract_strided_metadata %arg :
    memref<i4, strided<[], offset: ?>> -> memref<i4>, index
  return %offset : index
}

// -----

// Non-sub-byte source: pattern must not fire and the op is left for the
// existing `populateResolveExtractStridedMetadataPatterns` pipeline (which
// here resolves it on the unchanged i8 source).

// CHECK-LABEL: func.func @extract_strided_metadata_i8_passthrough
// CHECK-SAME:    %[[ARG:.+]]: memref<8xi8>
// CHECK:         %[[BASE:[A-Za-z0-9_]+]], %[[OFF:.+]], %[[SZ:.+]], %[[ST:.+]] = memref.extract_strided_metadata %[[ARG]] : memref<8xi8>
// CHECK-NOT:     arith.muli
// CHECK:         return %[[OFF]], %[[SZ]], %[[ST]]
func.func @extract_strided_metadata_i8_passthrough(%arg: memref<8xi8>)
    -> (index, index, index) {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %arg :
    memref<8xi8> -> memref<i8>, index, index, index
  return %offset, %size, %stride : index, index, index
}
