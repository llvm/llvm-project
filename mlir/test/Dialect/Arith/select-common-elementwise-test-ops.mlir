// RUN: mlir-opt %s --canonicalize="test-convergence" | FileCheck %s
// RUN: mlir-opt %s --canonicalize="test-convergence" --cse | FileCheck %s

// A shared scalar parameter is legal with an elementwise tensor mask.
// CHECK-LABEL: @shared_scalar
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : tensor<4xi1>, tensor<4xf32>
// CHECK-NEXT: %[[R:.*]] = "test.pure_elementwise"(%[[S]], %arg3) {callee = @foo} : (tensor<4xf32>, i32) -> tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @shared_scalar(%c: tensor<4xi1>, %a: tensor<4xf32>, %b: tensor<4xf32>, %k: i32) -> tensor<4xf32> {
  %x = "test.pure_elementwise"(%a, %k) {callee = @foo} : (tensor<4xf32>, i32) -> tensor<4xf32>
  %y = "test.pure_elementwise"(%b, %k) {callee = @foo} : (tensor<4xf32>, i32) -> tensor<4xf32>
  %r = arith.select %c, %x, %y : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}

// A tensor mask cannot select differing scalar parameters. Erase the select
// created for the first operand pair when the second pair fails validation.
// CHECK-LABEL: @different_scalars_tensor
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1, %arg3)
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2, %arg4)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : tensor<4xi1>, tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @different_scalars_tensor(%c: tensor<4xi1>, %a: tensor<4xf32>, %b: tensor<4xf32>, %k: i32, %l: i32) -> tensor<4xf32> {
  %x = "test.pure_elementwise"(%a, %k) : (tensor<4xf32>, i32) -> tensor<4xf32>
  %y = "test.pure_elementwise"(%b, %l) : (tensor<4xf32>, i32) -> tensor<4xf32>
  %r = arith.select %c, %x, %y : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: @different_scalars_vector
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1, %arg3)
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2, %arg4)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : vector<4xi1>, vector<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @different_scalars_vector(%c: vector<4xi1>, %a: vector<4xf32>, %b: vector<4xf32>, %k: i32, %l: i32) -> vector<4xf32> {
  %x = "test.pure_elementwise"(%a, %k) : (vector<4xf32>, i32) -> vector<4xf32>
  %y = "test.pure_elementwise"(%b, %l) : (vector<4xf32>, i32) -> vector<4xf32>
  %r = arith.select %c, %x, %y : vector<4xi1>, vector<4xf32>
  return %r : vector<4xf32>
}

// A scalar condition can select both the tensor and the scalar parameters.
// CHECK-LABEL: @scalar_condition_mixed_operands
// CHECK-NEXT: %[[S0:.*]] = arith.select %arg0, %arg1, %arg2 : tensor<4xf32>
// CHECK-NEXT: %[[S1:.*]] = arith.select %arg0, %arg3, %arg4 : i32
// CHECK-NEXT: %[[R:.*]] = "test.pure_elementwise"(%[[S0]], %[[S1]]) : (tensor<4xf32>, i32) -> tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @scalar_condition_mixed_operands(%c: i1, %a: tensor<4xf32>, %b: tensor<4xf32>, %k: i32, %l: i32) -> tensor<4xf32> {
  %x = "test.pure_elementwise"(%a, %k) : (tensor<4xf32>, i32) -> tensor<4xf32>
  %y = "test.pure_elementwise"(%b, %l) : (tensor<4xf32>, i32) -> tensor<4xf32>
  %r = arith.select %c, %x, %y : tensor<4xf32>
  return %r : tensor<4xf32>
}

// Elementwise permits compatible dynamic shapes; select requires exact types.
// CHECK-LABEL: @different_mask_shape
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1)
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : tensor<4xi1>, tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @different_mask_shape(%c: tensor<4xi1>, %a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<4xf32> {
  %x = "test.pure_elementwise"(%a) : (tensor<?xf32>) -> tensor<4xf32>
  %y = "test.pure_elementwise"(%b) : (tensor<?xf32>) -> tensor<4xf32>
  %r = arith.select %c, %x, %y : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}

// Matching dimensions alone are insufficient when encodings differ.
// CHECK-LABEL: @different_mask_encoding
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1)
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : tensor<4xi1, "out">, tensor<4xf32, "out">
// CHECK-NEXT: return %[[R]]
func.func @different_mask_encoding(%c: tensor<4xi1, "out">, %a: tensor<4xf32, "in">, %b: tensor<4xf32, "in">) -> tensor<4xf32, "out"> {
  %x = "test.pure_elementwise"(%a) : (tensor<4xf32, "in">) -> tensor<4xf32, "out">
  %y = "test.pure_elementwise"(%b) : (tensor<4xf32, "in">) -> tensor<4xf32, "out">
  %r = arith.select %c, %x, %y : tensor<4xi1, "out">, tensor<4xf32, "out">
  return %r : tensor<4xf32, "out">
}

// A shared input does not need to match the mask's encoding.
// CHECK-LABEL: @shared_different_encoding
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : tensor<4xi1, "out">, tensor<4xf32, "out">
// CHECK-NEXT: %[[R:.*]] = "test.pure_elementwise"(%[[S]], %arg3) : (tensor<4xf32, "out">, tensor<4xf32, "in">) -> tensor<4xf32, "out">
// CHECK-NEXT: return %[[R]]
func.func @shared_different_encoding(%c: tensor<4xi1, "out">, %a: tensor<4xf32, "out">, %b: tensor<4xf32, "out">, %k: tensor<4xf32, "in">) -> tensor<4xf32, "out"> {
  %x = "test.pure_elementwise"(%a, %k) : (tensor<4xf32, "out">, tensor<4xf32, "in">) -> tensor<4xf32, "out">
  %y = "test.pure_elementwise"(%b, %k) : (tensor<4xf32, "out">, tensor<4xf32, "in">) -> tensor<4xf32, "out">
  %r = arith.select %c, %x, %y : tensor<4xi1, "out">, tensor<4xf32, "out">
  return %r : tensor<4xf32, "out">
}

// CHECK-LABEL: @different_symbols
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1) {callee = @foo}
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2) {callee = @bar}
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @different_symbols(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = "test.pure_elementwise"(%a) {callee = @foo} : (f32) -> f32
  %y = "test.pure_elementwise"(%b) {callee = @bar} : (f32) -> f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Speculatability alone is insufficient, even for an elementwise operation.
// CHECK-LABEL: @elementwise_read
// CHECK-NEXT: %[[X:.*]] = "test.read_elementwise"(%arg1) {callee = @foo}
// CHECK-NEXT: %[[Y:.*]] = "test.read_elementwise"(%arg2) {callee = @foo}
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @elementwise_read(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = "test.read_elementwise"(%a) {callee = @foo} : (f32) -> f32
  %y = "test.read_elementwise"(%b) {callee = @foo} : (f32) -> f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Same operation name and result type do not imply matching operand types.
// CHECK-LABEL: @different_operand_types
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1) : (f16) -> f32
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2) : (bf16) -> f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @different_operand_types(%c: i1, %a: f16, %b: bf16) -> f32 {
  %x = "test.pure_elementwise"(%a) : (f16) -> f32
  %y = "test.pure_elementwise"(%b) : (bf16) -> f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Multiple results remain outside the pattern even if there is only one use.
// CHECK-LABEL: @multiple_results
// CHECK-NEXT: %[[X:.*]]:2 = "test.pure_elementwise"(%arg1) : (f32) -> (f32, f32)
// CHECK-NEXT: %[[Y:.*]]:2 = "test.pure_elementwise"(%arg2) : (f32) -> (f32, f32)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]]#0, %[[Y]]#0 : f32
// CHECK-NEXT: return %[[R]]
func.func @multiple_results(%c: i1, %a: f32, %b: f32) -> f32 {
  %x:2 = "test.pure_elementwise"(%a) : (f32) -> (f32, f32)
  %y:2 = "test.pure_elementwise"(%b) : (f32) -> (f32, f32)
  %r = arith.select %c, %x#0, %y#0 : f32
  return %r : f32
}

// Corresponding operand encodings must match even with a scalar condition.
// CHECK-LABEL: @different_operand_encodings
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1)
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : tensor<4xf32, "out">
// CHECK-NEXT: return %[[R]]
func.func @different_operand_encodings(%c: i1, %a: tensor<4xf32, "a">, %b: tensor<4xf32, "b">) -> tensor<4xf32, "out"> {
  %x = "test.pure_elementwise"(%a) : (tensor<4xf32, "a">) -> tensor<4xf32, "out">
  %y = "test.pure_elementwise"(%b) : (tensor<4xf32, "b">) -> tensor<4xf32, "out">
  %r = arith.select %c, %x, %y : tensor<4xf32, "out">
  return %r : tensor<4xf32, "out">
}

// Comparing regions while ignoring SSA values would consider these bodies
// equivalent. Replacing only the top-level operands cannot select k0 versus k1:
// cloning the true producer would retain k0 even when c is false.
// CHECK-LABEL: @different_region_captures
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise_region"(%arg1) ({
// CHECK-NEXT: ^bb0(%[[A:.*]]: f32):
// CHECK-NEXT: %[[SUM0:.*]] = arith.addf %[[A]], %arg3 : f32
// CHECK-NEXT: "test.return"(%[[SUM0]]) : (f32) -> ()
// CHECK-NEXT: }) : (f32) -> f32
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise_region"(%arg2) ({
// CHECK-NEXT: ^bb0(%[[B:.*]]: f32):
// CHECK-NEXT: %[[SUM1:.*]] = arith.addf %[[B]], %arg4 : f32
// CHECK-NEXT: "test.return"(%[[SUM1]]) : (f32) -> ()
// CHECK-NEXT: }) : (f32) -> f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @different_region_captures(%c: i1, %a: f32, %b: f32, %k0: f32, %k1: f32) -> f32 {
  %x = "test.pure_elementwise_region"(%a) ({
  ^bb0(%v: f32):
    %sum = arith.addf %v, %k0 : f32
    "test.return"(%sum) : (f32) -> ()
  }) : (f32) -> f32
  %y = "test.pure_elementwise_region"(%b) ({
  ^bb0(%v: f32):
    %sum = arith.addf %v, %k1 : f32
    "test.return"(%sum) : (f32) -> ()
  }) : (f32) -> f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Failure after creating two selects must erase each exactly once, even though
// the first is reused. Keep existing operands and their defining operations.
// CHECK-LABEL: @cleanup_partial_operands
// CHECK-NEXT: %[[A:.*]] = math.sqrt %arg1 : tensor<4xf32>
// CHECK-NEXT: %[[B:.*]] = math.sqrt %arg2 : tensor<4xf32>
// CHECK-NEXT: %[[SHARED:.*]] = math.sqrt %arg5 : tensor<4xf32>
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%[[SHARED]], %[[A]], %[[A]], %arg3, %arg6)
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%[[SHARED]], %[[B]], %[[B]], %arg4, %arg7)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : tensor<4xi1>, tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @cleanup_partial_operands(%c: tensor<4xi1>, %a_in: tensor<4xf32>, %b_in: tensor<4xf32>, %d: tensor<4xf32>, %e: tensor<4xf32>, %shared_in: tensor<4xf32>, %k: i32, %l: i32) -> tensor<4xf32> {
  %a = math.sqrt %a_in : tensor<4xf32>
  %b = math.sqrt %b_in : tensor<4xf32>
  %shared = math.sqrt %shared_in : tensor<4xf32>
  %x = "test.pure_elementwise"(%shared, %a, %a, %d, %k) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, i32) -> tensor<4xf32>
  %y = "test.pure_elementwise"(%shared, %b, %b, %e, %l) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, i32) -> tensor<4xf32>
  %r = arith.select %c, %x, %y : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}

// An encoding mismatch on a later pair must also clean up earlier selects.
// CHECK-LABEL: @cleanup_on_encoding_mismatch
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise"(%arg1, %arg3)
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise"(%arg2, %arg4)
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : tensor<4xi1, "out">, tensor<4xf32, "out">
// CHECK-NEXT: return %[[R]]
func.func @cleanup_on_encoding_mismatch(%c: tensor<4xi1, "out">, %a: tensor<4xf32, "out">, %b: tensor<4xf32, "out">, %k: tensor<4xf32, "in">, %l: tensor<4xf32, "in">) -> tensor<4xf32, "out"> {
  %x = "test.pure_elementwise"(%a, %k) : (tensor<4xf32, "out">, tensor<4xf32, "in">) -> tensor<4xf32, "out">
  %y = "test.pure_elementwise"(%b, %l) : (tensor<4xf32, "out">, tensor<4xf32, "in">) -> tensor<4xf32, "out">
  %r = arith.select %c, %x, %y : tensor<4xi1, "out">, tensor<4xf32, "out">
  return %r : tensor<4xf32, "out">
}
