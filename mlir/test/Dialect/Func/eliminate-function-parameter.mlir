// RUN: mlir-opt %s --split-input-file --eliminate-function-parameter | \
// RUN: FileCheck %s

func.func @single_parameter(%arg: index) {
  return
}

func.func @mutl_parameter(%arg0 : index, %arg1 : index) -> index {
  return %arg0 : index
}

func.func @eliminate_parameter(%arg0: index, %arg1: index) -> index {
  func.call @single_parameter(%arg0) : (index) -> ()
  %ret = func.call @mutl_parameter(%arg0, %arg0) : (index, index) -> (index)
  return %ret : index
}

// CHECK-LABEL: func @single_parameter() {
//       CHECK:   return
//       CHECK: }

// CHECK-LABEL: func @mutl_parameter(
//  CHECK-SAME:   %[[ARG0:.*]]: index) -> index {
//       CHECK:   return %[[ARG0]] : index
//       CHECK: }

// CHECK-LABEL: func @eliminate_parameter(
//  CHECK-SAME:   %[[ARG0:.*]]: index) -> index {
//       CHECK:   call @single_parameter() : () -> ()
//       CHECK:   %[[RET:.*]] = call @mutl_parameter(%[[ARG0]]) : (index) -> index
//       CHECK:   return %[[RET]] : index
//       CHECK: }
