// RUN: mlir-opt --arith-int-range-narrowing="int-bitwidths-supported=1,8,16,32" %s | FileCheck %s

// Test that the pass doesn't crash on operations that implement
// LoopLikeOpInterface but don't provide loop bounds (e.g., tensor.generate).
// See https://github.com/llvm/llvm-project/issues/180312

// CHECK-LABEL: func @tensor_generate_no_crash
func.func @tensor_generate_no_crash(%arg0: index) -> tensor<?xf32> {
  %cst = arith.constant 1.0 : f32
  // tensor.generate implements LoopLikeOpInterface but getLoopLowerBounds(),
  // getLoopUpperBounds(), and getLoopSteps() return nullopt.
  %0 = tensor.generate %arg0 {
  ^bb0(%i: index):
    tensor.yield %cst : f32
  } : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @tensor_generate_with_arith
func.func @tensor_generate_with_arith(%arg0: index) -> tensor<?xindex> {
  %c1 = arith.constant 1 : index
  %0 = tensor.generate %arg0 {
  ^bb0(%i: index):
    %sum = arith.addi %i, %c1 : index
    tensor.yield %sum : index
  } : tensor<?xindex>
  return %0 : tensor<?xindex>
}
