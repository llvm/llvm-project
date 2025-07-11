// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// -----

// CHECK-LABEL: test_cond_if_generic_form
// CHECK: %[[OUT:.*]] = tosa.cond_if(%[[COND:.*]], %[[IN0:.*]], %[[IN1:.*]]) ({
// CHECK: ^bb0(%[[INA:.*]]: tensor<f32>, %[[INB:.*]]: tensor<f32>):
// CHECK:    %[[THEN_TERM:.*]] = tosa.add %[[INA]], %[[INB]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:    tosa.yield %[[THEN_TERM]] : tensor<f32>
// CHECK: }, {
// CHECK: ^bb0(%[[INC:.*]]: tensor<f32>, %[[IND:.*]]: tensor<f32>):
// CHECK:    %[[ELSE_TERM:.*]] = tosa.sub %[[INC]], %[[IND]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:    tosa.yield %[[ELSE_TERM]] : tensor<f32>
// CHECK: }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: return %[[OUT]] : tensor<f32>
func.func @test_cond_if_generic_form(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> tensor<f32> {
  %0 = tosa.cond_if(%arg2, %arg0, %arg1) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = tosa.add %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = tosa.sub %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_cond_if_generic_form_no_block_arguments
// COM: No block arguments means simplified form can be printed
func.func @test_cond_if_generic_form_no_block_arguments(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> tensor<f32> {
  // CHECK: tosa.cond_if %arg2 -> (tensor<f32>)
  %0 = tosa.cond_if(%arg2) ({
  ^bb0():
    tosa.yield %arg0 : tensor<f32>
  },  {
  ^bb0():
    tosa.yield %arg1 : tensor<f32>
  }) : (tensor<i1>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_cond_if_simplified_form
// CHECK: tosa.cond_if %arg2 -> (tensor<f32>)
func.func @test_cond_if_simplified_form(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> tensor<f32> {
  %0 = tosa.cond_if %arg2 -> (tensor<f32>) {
    %1 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  } else {
    %1 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_cond_if_simplified_form_just_yield
// CHECK: tosa.cond_if %arg2 -> (tensor<f32>)
func.func @test_cond_if_simplified_form_just_yield(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> tensor<f32> {
  %0 = tosa.cond_if %arg2 -> (tensor<f32>) {
    tosa.yield %arg0 : tensor<f32>
  } else {
    tosa.yield %arg1 : tensor<f32>
  }
  return %0 : tensor<f32>
}
