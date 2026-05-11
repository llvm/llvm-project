// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8 enable-cf-conversion=true" --cse --verify-diagnostics --split-input-file %s | FileCheck %s
// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8 enable-cf-conversion=true" --split-input-file %s | FileCheck %s --check-prefix=CHECK-NOCSE

// Sub-byte memref type carried through cf.br block args. The
// BranchOpInterface type-conversion pattern must rewrite both the cf.br
// operand type and the successor block-arg type to the i8 container, so the
// downstream uses in the successor block see an i8 source.

// CHECK-LABEL: func.func @cf_br_block_arg_narrow_type
// CHECK-SAME:    %[[ARG:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>
// CHECK:         cf.br ^[[BB1:.+]](%[[ARG]] : memref<{{[0-9]+}}xi8>)
// CHECK:       ^[[BB1]](%[[BARG:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>):
// CHECK:         return %[[BARG]]
// CHECK-NOT:     memref<{{[0-9]+}}xi4>
func.func @cf_br_block_arg_narrow_type(%arg: memref<8xi4>) -> memref<8xi4> {
  cf.br ^bb1(%arg : memref<8xi4>)
^bb1(%a: memref<8xi4>):
  return %a : memref<8xi4>
}

// -----

// Sub-byte memref with dynamic offset carried through cf.br block-arg,
// then loaded via vector.load. After FunctionOpInterfaceAllBlocksSignatureConversion
// converts the block-arg to i8, an unrealized_conversion_cast is inserted.
// ConvertVectorLoad calls extract_strided_metadata on the original sub-byte
// source (op.getBase()), which is illegal without ConvertExtractStridedMetadata.

// CHECK-LABEL: func.func @cf_br_block_arg_vector_load_i4
// CHECK:         vector.load {{.*}} : memref<{{[0-9]+}}xi8, strided<[1], offset: ?>>, vector<{{[0-9]+}}xi8>
// CHECK-NOT:     memref<{{[0-9]+}}xi4>
// CHECK-NOCSE-LABEL: func.func @cf_br_block_arg_vector_load_i4
// CHECK-NOCSE-NOT:   memref<{{[0-9]+}}xi4>
func.func @cf_br_block_arg_vector_load_i4(%arg: memref<8xi4, strided<[1], offset: ?>>) -> vector<8xi4> {
  cf.br ^bb1(%arg : memref<8xi4, strided<[1], offset: ?>>)
^bb1(%a: memref<8xi4, strided<[1], offset: ?>>):
  %c0 = arith.constant 0 : index
  %v = vector.load %a[%c0] : memref<8xi4, strided<[1], offset: ?>>, vector<8xi4>
  return %v : vector<8xi4>
}

// -----

// Sub-byte memref with static non-zero offset carried through cf.br block-arg,
// then loaded via vector.load. The static offset (4 i4-elements) is converted
// to 2 i8-elements in the container type. ConvertExtractStridedMetadata must
// NOT over-scale the static offset: srcStaticOffset is already in
// emulated-element (i4) units, so it should be passed through unchanged rather
// than multiplied by containerBits/emulatedBits (which would give 8, wrong).
//
// The CHECK-NOCSE check verifies that the emulated offset constant is 4 (in i4
// units, as returned by ConvertExtractStridedMetadata) and not 8 (the
// over-scaled value that would appear with the bug).  After CSE the constant
// is dead and the distinction disappears, so this check requires the no-CSE
// run line.

// CHECK-LABEL: func.func @cf_br_block_arg_vector_load_i4_static_offset
// CHECK:         vector.load {{.*}} : memref<{{[0-9]+}}xi8, strided<[1], offset: 2>>, vector<{{[0-9]+}}xi8>
// CHECK-NOT:     memref<{{[0-9]+}}xi4>
// CHECK-NOCSE-LABEL: func.func @cf_br_block_arg_vector_load_i4_static_offset
// CHECK-NOCSE-NOT:   memref<{{[0-9]+}}xi4>
// CHECK-NOCSE:       %c4 = arith.constant 4 : index
func.func @cf_br_block_arg_vector_load_i4_static_offset(%arg: memref<8xi4, strided<[1], offset: 4>>) -> vector<4xi4> {
  cf.br ^bb1(%arg : memref<8xi4, strided<[1], offset: 4>>)
^bb1(%a: memref<8xi4, strided<[1], offset: 4>>):
  %c0 = arith.constant 0 : index
  %v = vector.load %a[%c0] : memref<8xi4, strided<[1], offset: 4>>, vector<4xi4>
  return %v : vector<4xi4>
}
