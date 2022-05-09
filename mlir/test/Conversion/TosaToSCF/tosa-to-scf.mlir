// RUN: mlir-opt --split-input-file --tosa-to-scf %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @while_test
// CHECK-SAME: ([[ARG0:%.+]]: tensor<i32>)
func.func @while_test(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // CHECK: [[WHILE:%.+]] = scf.while ([[ARG1:%.+]] = [[ARG0]])
  %1 = "tosa.while_loop"(%arg0) ({
  ^bb0(%arg2: tensor<i32>):
    // CHECK: "tosa.const"
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>

    // CHECK: [[COMPARE:%.+]] = "tosa.greater_equal"
    %3 = "tosa.greater_equal"(%2, %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i1>

    // CHECK: [[EX:%.+]] = tensor.extract [[COMPARE]]
    // CHECK: scf.condition([[EX]]) [[ARG1]]
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  // CHECK: ^bb0([[ARG1:%.+]]: tensor<i32>)
  ^bb0(%arg2: tensor<i32>):
    // CHECK: tosa.const
    %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

    // CHECK: [[ADD:%.+]] = "tosa.add"
    %3 = "tosa.add"(%arg2, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>

    // CHECK: scf.yield [[ADD]]
    "tosa.yield"(%3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>)
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: func @if_test
// CHECK-SAME: ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<i1>)
func.func @if_test(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> (tensor<f32>) {
  // CHECK: [[EX:%.+]] = tensor.extract [[ARG2]]
  // CHECK: [[IF:%.+]] = scf.if [[EX]] -> (tensor<f32>) {
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({

  // CHECK:   scf.yield [[ARG0]]
  ^bb1(%arg3 : tensor<f32>, %arg4 : tensor<f32>):
    "tosa.yield"(%arg3) : (tensor<f32>) -> ()

  // CHECK: } else {
  }, {

  // CHECK:   scf.yield [[ARG1]]
  ^bb1(%arg5 : tensor<f32>, %arg6 : tensor<f32>):
    "tosa.yield"(%arg6) : (tensor<f32>) -> ()

  // CHECK: }
  // CHECK: return [[IF]]
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> (tensor<f32>)

  return %0 : tensor<f32>
}
