// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8 arith-compute-bitwidth=1" --cse --verify-diagnostics --split-input-file %s | FileCheck %s

// Sub-byte memref type carried through cf.br block args. The cf branch
// pattern (registered by cf::populateCFStructuralTypeConversionsAndLegality)
// must rewrite both the cf.br operand type and the successor block-arg type
// to the i8 container, so the downstream uses in the successor block see an
// i8 source.

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

// Sub-byte memref carried through both successors of a cf.cond_br. Both
// branch operand types and both successor block-arg types must be rewritten
// to the i8 container.

// CHECK-LABEL: func.func @cf_cond_br_block_arg_narrow_type
// CHECK-SAME:    %[[COND:[A-Za-z0-9_]+]]: i1
// CHECK-SAME:    %[[A:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>
// CHECK-SAME:    %[[B:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>
// CHECK:         cf.cond_br %[[COND]], ^[[BBT:.+]](%[[A]] : memref<{{[0-9]+}}xi8>), ^[[BBF:.+]](%[[B]] : memref<{{[0-9]+}}xi8>)
// CHECK:       ^[[BBT]](%[[XT:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>):
// CHECK:         return %[[XT]]
// CHECK:       ^[[BBF]](%[[XF:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>):
// CHECK:         return %[[XF]]
// CHECK-NOT:     memref<{{[0-9]+}}xi4>
func.func @cf_cond_br_block_arg_narrow_type(%cond: i1, %a: memref<8xi4>, %b: memref<8xi4>) -> memref<8xi4> {
  cf.cond_br %cond, ^bb1(%a : memref<8xi4>), ^bb2(%b : memref<8xi4>)
^bb1(%x: memref<8xi4>):
  return %x : memref<8xi4>
^bb2(%y: memref<8xi4>):
  return %y : memref<8xi4>
}

// -----

// Sub-byte memref carried through the default and case successors of a
// cf.switch. The branch pattern must rewrite the operand type at every
// successor edge and the matching block-arg type at every successor.

// CHECK-LABEL: func.func @cf_switch_block_arg_narrow_type
// CHECK-SAME:    %[[FLAG:[A-Za-z0-9_]+]]: i32
// CHECK-SAME:    %[[ARG:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>
// CHECK:         cf.switch %[[FLAG]] : i32, [
// CHECK:           default: ^[[BBD:.+]](%[[ARG]] : memref<{{[0-9]+}}xi8>)
// CHECK:           0: ^[[BB0:.+]](%[[ARG]] : memref<{{[0-9]+}}xi8>)
// CHECK:           1: ^[[BB1:.+]](%[[ARG]] : memref<{{[0-9]+}}xi8>)
// CHECK:         ]
// CHECK:       ^[[BBD]](%[[XD:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>):
// CHECK:         return %[[XD]]
// CHECK:       ^[[BB0]](%[[X0:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>):
// CHECK:         return %[[X0]]
// CHECK:       ^[[BB1]](%[[X1:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>):
// CHECK:         return %[[X1]]
// CHECK-NOT:     memref<{{[0-9]+}}xi4>
func.func @cf_switch_block_arg_narrow_type(%flag: i32, %arg: memref<8xi4>) -> memref<8xi4> {
  cf.switch %flag : i32, [
    default: ^bb1(%arg : memref<8xi4>),
    0: ^bb2(%arg : memref<8xi4>),
    1: ^bb3(%arg : memref<8xi4>)
  ]
^bb1(%x: memref<8xi4>):
  return %x : memref<8xi4>
^bb2(%y: memref<8xi4>):
  return %y : memref<8xi4>
^bb3(%z: memref<8xi4>):
  return %z : memref<8xi4>
}
