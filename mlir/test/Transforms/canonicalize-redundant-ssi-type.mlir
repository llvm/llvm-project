// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}))' \
// RUN:   | FileCheck %s

// Verify that a block argument with SSIType is NOT removed by
// dropRedundantArguments, even when all predecessors pass the same value.
// Uses "test.br" (not cf.br) so block merging doesn't obscure the result.

// CHECK-LABEL: func @redundant_ssi_arg_preserved
// CHECK:         "test.br"(%{{.*}})[^bb1] : (!test.ssi_type) -> ()
// CHECK:       ^bb1(%[[V:.*]]: !test.ssi_type):
// CHECK-NEXT:    %[[W:.*]] = "test.use"(%[[V]])
// CHECK-NEXT:    return %[[W]]
func.func @redundant_ssi_arg_preserved(%arg0: !test.ssi_type) -> !test.ssi_type {
  "test.br"(%arg0)[^succ] : (!test.ssi_type) -> ()
^succ(%0: !test.ssi_type):
  %1 = "test.use"(%0) : (!test.ssi_type) -> !test.ssi_type
  return %1 : !test.ssi_type
}

// -----

// Verify that a dead SSI block argument IS removed -- deadness is safe to
// fold; only collapsing a live arg across control-flow paths is forbidden.
// Uses "test.br" so block merging doesn't obscure the result.

// CHECK-LABEL: func @dead_ssi_arg_eliminated
// CHECK:         "test.br"()[^bb1]
// CHECK:       ^bb1:
// CHECK-NEXT:    return
func.func @dead_ssi_arg_eliminated(%arg0: !test.ssi_type) {
  "test.br"(%arg0)[^succ] : (!test.ssi_type) -> ()
^succ(%0: !test.ssi_type):
  return
}

// -----

// Verify that a normal (non-SSI) redundant block argument IS removed by
// dropRedundantArguments, confirming the baseline behavior we are opting out
// of.

// CHECK-LABEL: func @redundant_normal_arg_eliminated
// CHECK-NEXT:    return
func.func @redundant_normal_arg_eliminated(%arg0: f32) -> f32 {
  cf.br ^succ(%arg0 : f32)
^succ(%0: f32):
  return %0 : f32
}

// -----

// Verify that SSI and non-SSI args can coexist: the non-SSI dead arg is
// eliminated while the SSI arg is preserved.

// CHECK-LABEL: func @mixed_args_selective_preservation
// CHECK:         "test.br"(%{{.*}})[^bb1]
// CHECK:       ^bb1(%[[V:.*]]: !test.ssi_type):
// CHECK-NEXT:    %[[W:.*]] = "test.use"(%[[V]])
// CHECK-NEXT:    return %[[W]]
func.func @mixed_args_selective_preservation(%a: !test.ssi_type, %b: f32) -> (!test.ssi_type, f32){
  "test.br"(%a, %b)[^succ] : (!test.ssi_type, f32) -> ()
^succ(%0: !test.ssi_type, %1: f32):
  %2 = "test.use"(%0) : (!test.ssi_type) -> !test.ssi_type
  return %2, %1 : !test.ssi_type, f32
}

// -----

// CHECK-LABEL: func @ssi_preserved_block_merged(
// CHECK-SAME:      %[[V:.*]]: !test.ssi_type)
// CHECK-NEXT:    %[[W:.*]] = "test.use"(%[[V]])
// CHECK-NEXT:    return %[[W]]
func.func @ssi_preserved_block_merged(%arg0: !test.ssi_type) -> !test.ssi_type {
  cf.br ^succ(%arg0 : !test.ssi_type)
^succ(%0: !test.ssi_type):
  %1 = "test.use"(%0) : (!test.ssi_type) -> !test.ssi_type
  return %1 : !test.ssi_type
}

// -----

// CHECK-LABEL:   func.func @ssi_preserved_cond_br(
// CHECK-SAME:      %[[ARG0:.*]]: i1,
// CHECK-SAME:      %[[ARG1:.*]]: !test.ssi_type)
// CHECK:           cf.cond_br %[[ARG0]], ^bb1(%[[ARG1]] : !test.ssi_type), ^bb2(%[[ARG1]] : !test.ssi_type)
// CHECK:         ^bb1(%[[VAL_0:[^:]*]]:
// CHECK:           %[[VAL_1:.*]] = "test.use1"(%[[VAL_0]])
// CHECK:           cf.br ^bb3(%[[VAL_1]]
// CHECK:         ^bb2(%[[VAL_2:[^:]*]]:
// CHECK:           %[[VAL_3:.*]] = "test.use2"(%[[VAL_2]])
// CHECK:           cf.br ^bb3(%[[VAL_3]]
// CHECK:         ^bb3(%[[VAL_4:[^:]*]]:
// CHECK:           return %[[VAL_4]]
func.func @ssi_preserved_cond_br(%arg0: i1, %arg1: !test.ssi_type) -> !test.ssi_type {
  cf.cond_br %arg0, ^succ0(%arg1 : !test.ssi_type), ^succ1(%arg1 : !test.ssi_type)
^succ0(%0: !test.ssi_type):
  %1 = "test.use1"(%0) : (!test.ssi_type) -> !test.ssi_type
  cf.br ^exit(%1 : !test.ssi_type)
^succ1(%2: !test.ssi_type):
  %3 = "test.use2"(%2) : (!test.ssi_type) -> !test.ssi_type
  cf.br ^exit(%3 : !test.ssi_type)
^exit(%4: !test.ssi_type):
  return %4 : !test.ssi_type
}
