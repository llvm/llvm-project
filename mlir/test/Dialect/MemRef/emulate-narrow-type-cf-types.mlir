// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8 arith-compute-bitwidth=8" --cse --verify-diagnostics --split-input-file %s | FileCheck %s

// Sub-byte integer scalar carried through a cf.br block-arg. The cf branch
// pattern (registered via cf::populateCFStructuralTypeConversionsAndLegality
// in the memref / vector narrow-type emulation wrappers) rewrites both the
// cf.br operand type and the successor block-arg type using the arith
// narrow-type integer converter.

// CHECK-LABEL: func.func @cf_br_block_arg_arith_i4
// CHECK-SAME:    %[[ARG:[A-Za-z0-9_]+]]: i8
// CHECK:         cf.br ^[[BB1:.+]](%[[ARG]] : i8)
// CHECK:       ^[[BB1]](%[[BARG:[A-Za-z0-9_]+]]: i8):
// CHECK:         return %[[BARG]]
// CHECK-NOT:     i4
func.func @cf_br_block_arg_arith_i4(%arg: i4) -> i4 {
  cf.br ^bb1(%arg : i4)
^bb1(%a: i4):
  return %a : i4
}

// -----

// Sub-byte integer vector carried through a cf.br block-arg.
// cf::populateCFStructuralTypeConversionsAndLegality rewrites both the
// cf.br operand type and the successor block-arg type using the arith
// narrow-type vector converter.

// CHECK-LABEL: func.func @cf_br_block_arg_vector_i4
// CHECK-SAME:    %[[ARG:[A-Za-z0-9_]+]]: vector<8xi8>
// CHECK:         cf.br ^[[BB1:.+]](%[[ARG]] : vector<8xi8>)
// CHECK:       ^[[BB1]](%[[BARG:[A-Za-z0-9_]+]]: vector<8xi8>):
// CHECK:         return %[[BARG]]
// CHECK-NOT:     vector<8xi4>
func.func @cf_br_block_arg_vector_i4(%arg: vector<8xi4>) -> vector<8xi4> {
  cf.br ^bb1(%arg : vector<8xi4>)
^bb1(%a: vector<8xi4>):
  return %a : vector<8xi4>
}
