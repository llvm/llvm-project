// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}))' \
// RUN:   | FileCheck %s

// Verify that a block argument with SSIType is NOT removed by
// dropRedundantArguments, even when all predecessors pass the same value.

// CHECK-LABEL: func @redundant_ssi_arg_preserved
// CHECK:         cf.br ^bb1(%{{.*}} : !test.ssi_type)
// CHECK:       ^bb1(%{{.*}}: !test.ssi_type):
// CHECK-NEXT:    return
func.func @redundant_ssi_arg_preserved(%arg0: !test.ssi_type) {
  cf.br ^succ(%arg0 : !test.ssi_type)
^succ(%0: !test.ssi_type):
  return
}

// -----

// Verify that a normal (non-SSI) redundant block argument IS removed by
// dropRedundantArguments, confirming the baseline behavior we are opting out
// of.

// CHECK-LABEL: func @redundant_normal_arg_eliminated
// CHECK:         cf.br ^bb1
// CHECK:       ^bb1:
// CHECK-NEXT:    return
func.func @redundant_normal_arg_eliminated(%arg0: f32) {
  cf.br ^succ(%arg0 : f32)
^succ(%0: f32):
  return
}

// -----

// Verify that SSI and non-SSI args can coexist: the non-SSI dead arg is
// eliminated while the SSI arg is preserved.

// CHECK-LABEL: func @mixed_args_selective_preservation
// CHECK:         "test.br"(%{{.*}})[^bb1]
// CHECK:       ^bb1(%{{.*}}: !test.ssi_type):
// CHECK-NEXT:    return
func.func @mixed_args_selective_preservation(%a: !test.ssi_type, %b: f32) {
  "test.br"(%a, %b)[^succ] : (!test.ssi_type, f32) -> ()
^succ(%0: !test.ssi_type, %1: f32):
  return
}
