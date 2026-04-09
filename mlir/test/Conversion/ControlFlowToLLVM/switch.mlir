// RUN: mlir-opt %s -convert-cf-to-llvm -split-input-file | FileCheck %s

// Unstructured control flow is converted, but the enclosing op is not
// converted.

// CHECK-LABEL: func.func @single_case(
//  CHECK-SAME:     %[[val:.*]]: i32, %[[idx:.*]]: index) -> index {
//       CHECK:   %[[cast0:.*]] = builtin.unrealized_conversion_cast %[[idx]] : index to i64
//       CHECK:   llvm.switch %[[val]] : i32, ^[[bb1:.*]](%[[cast0]] : i64) [
//       CHECK:   ]
//       CHECK: ^[[bb1]](%[[arg0:.*]]: i64):
//       CHECK:   %[[cast1:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : i64 to index
//       CHECK:   return %[[cast1]] : index
//       CHECK: }
func.func @single_case(%val: i32, %idx: index) -> index {
  cf.switch %val : i32, [
    default: ^bb1(%idx : index)
  ]
^bb1(%arg0: index):
  return %arg0 : index
}

// -----

// func.func and func.return types match. No unrealized_conversion_cast is
// needed.

// CHECK-LABEL: func.func @single_case_type_match(
//  CHECK-SAME:     %[[val:.*]]: i32, %[[i:.*]]: i64) -> i64 {
//       CHECK:   llvm.switch %[[val]] : i32, ^[[bb1:.*]](%[[i]] : i64) [
//       CHECK:   ]
//       CHECK: ^[[bb1]](%[[arg0:.*]]: i64):
//       CHECK:   return %[[arg0]] : i64
//       CHECK: }
func.func @single_case_type_match(%val: i32, %i: i64) -> i64 {
  cf.switch %val : i32, [
    default: ^bb1(%i : i64)
  ]
^bb1(%arg0: i64):
  return %arg0 : i64
}

// -----

//   CHECK-LABEL: func.func @multi_case
// CHECK-COUNT-2:   unrealized_conversion_cast {{.*}} : index to i64
//         CHECK:   llvm.switch %{{.*}} : i32, ^{{.*}}(%{{.*}} : i64) [
//         CHECK:     12: ^{{.*}}(%{{.*}} : i64),
//         CHECK:     13: ^{{.*}}(%{{.*}} : i64),
//         CHECK:     14: ^{{.*}}(%{{.*}} : i64)
//         CHECK:   ]
func.func @multi_case(%val: i32, %idx1: index, %idx2: index, %i: i64) -> index {
  cf.switch %val : i32, [
    default: ^bb1(%idx1 : index),
    12: ^bb2(%idx2 : index),
    13: ^bb1(%idx1 : index),
    14: ^bb3(%i : i64)
  ]
^bb1(%arg0: index):
  return %arg0 : index
^bb2(%arg1: index):
  return %arg1 : index
^bb3(%arg2: i64):
  %cast = arith.index_cast %arg2 : i64 to index
  return %cast : index
}
