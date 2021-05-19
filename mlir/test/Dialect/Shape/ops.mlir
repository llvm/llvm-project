// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: shape_num_elements
func @shape_num_elements(%shape : !shape.shape) -> !shape.size {
  %init = shape.const_size 1
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index : index, %extent : !shape.size, %acc : !shape.size):
      %acc_next = shape.mul %acc, %extent
          : !shape.size, !shape.size -> !shape.size
      shape.yield %acc_next : !shape.size
  }
  return %num_elements : !shape.size
}

// CHECK-LABEL: extent_tensor_num_elements
func @extent_tensor_num_elements(%shape : tensor<?xindex>) -> index {
  %init = constant 1 : index
  %num_elements = shape.reduce(%shape, %init) : tensor<?xindex> -> index {
    ^bb0(%index : index, %extent : index, %acc : index):
      %acc_next = shape.mul %acc, %extent : index, index -> index
      shape.yield %acc_next : index
  }
  return %num_elements : index
}

func @test_shape_num_elements_unknown() {
  %0 = "shape.unknown_shape"() : () -> !shape.shape
  %1 = call @shape_num_elements(%0) : (!shape.shape) -> (!shape.size)
  %2 = "shape.print"(%1) : (!shape.size) -> !shape.size
  return
}

func @const_shape() {
  %0 = shape.const_shape [1, 2, 3] : !shape.shape
  %1 = shape.const_shape [4, 5, 6] : tensor<?xindex>
  %2 = shape.const_shape [4, 5, 6] : tensor<3xindex>
  return
}

func @test_shape_num_elements_fixed() {
  %0 = shape.const_shape [1, 57, 92] : !shape.shape
  %1 = call @shape_num_elements(%0) : (!shape.shape) -> (!shape.size)
  %3 = "shape.print"(%1) : (!shape.size) -> !shape.size
  return
}

func @test_broadcast_fixed() {
  %0 = shape.const_shape [10, 1, 57, 92] : !shape.shape
  %1 = shape.const_shape [4, 57, 92] : !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_broadcast_extents() -> tensor<?xindex> {
  %0 = shape.const_shape [10, 1, 57, 92] : tensor<?xindex>
  %1 = shape.const_shape [4, 57, 92] : tensor<?xindex>
  %2 = shape.broadcast %0, %1 : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return %2 : tensor<?xindex>
}

func @test_shape_any_fixed() {
  %0 = shape.const_shape [4, 57, 92] : !shape.shape
  %1 = shape.const_shape [4, 57, 92] : !shape.shape
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_unknown() {
  %0 = shape.const_shape [4, -1, 92] : !shape.shape
  %1 = shape.const_shape [-1, 57, 92] : !shape.shape
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_fixed_mismatch() {
  %0 = shape.const_shape [4, 57, 92] : !shape.shape
  %1 = shape.const_shape [2, 57, 92] : !shape.shape
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_parse_const_shape() {
  %0 = shape.const_shape [] : !shape.shape
  %1 = shape.const_shape [1, 2, 3] : !shape.shape
  %2 = shape.const_shape [1, 2, 3] : tensor<?xindex>
  return
}

func @test_shape_of(%arg0: tensor<?xf32>) -> tensor<?xindex> {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  return %0 : tensor<?xindex>
}

func @test_constraints() {
  %0 = shape.const_shape [] : !shape.shape
  %1 = shape.const_shape [1, 2, 3] : !shape.shape
  %true = constant true
  %w0 = shape.cstr_broadcastable %0, %1 : !shape.shape, !shape.shape
  %w1 = shape.cstr_eq %0, %1 : !shape.shape, !shape.shape
  %w2 = shape.const_witness true
  %w3 = shape.const_witness false
  %w4 = shape.cstr_require %true, "msg"
  %w_all = shape.assuming_all %w0, %w1, %w2, %w3, %w4
  shape.assuming %w_all -> !shape.shape {
    %2 = "shape.any"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
    shape.assuming_yield %2 : !shape.shape
  }
  return
}

func @eq_on_extent_tensors(%lhs : tensor<?xindex>,
                           %rhs : tensor<?xindex>) {
  %w0 = shape.cstr_eq %lhs, %rhs : tensor<?xindex>, tensor<?xindex>
  return
}

func @broadcastable_on_extent_tensors(%lhs : tensor<?xindex>,
                                      %rhs : tensor<?xindex>) {
  %w0 = shape.cstr_broadcastable %lhs, %rhs : tensor<?xindex>, tensor<?xindex>
  return
}

func @mul(%size_arg : !shape.size, %index_arg : index) {
  %size_prod = shape.mul %size_arg, %size_arg
      : !shape.size, !shape.size -> !shape.size
  %index_prod = shape.mul %index_arg, %index_arg : index, index -> index
  %mixed_prod = shape.mul %size_arg, %index_arg
      : !shape.size, index -> !shape.size
  return
}

func @div(%size_arg : !shape.size, %index_arg : index) {
  %size_div = shape.div %size_arg, %size_arg
      : !shape.size, !shape.size -> !shape.size
  %index_div = shape.div %index_arg, %index_arg : index, index -> index
  %mixed_div = shape.div %size_arg, %index_arg
      : !shape.size, index -> !shape.size
  return
}

func @add(%size_arg : !shape.size, %index_arg : index) {
  %size_sum = shape.add %size_arg, %size_arg
      : !shape.size, !shape.size -> !shape.size
  %index_sum = shape.add %index_arg, %index_arg : index, index -> index
  %mixed_sum = shape.add %size_arg, %index_arg
      : !shape.size, index -> !shape.size
  return
}

func @const_size() {
  // CHECK: %c1 = shape.const_size 1
  // CHECK: %c2 = shape.const_size 2
  // CHECK: %c2_0 = shape.const_size 2
  %0 = shape.const_size 1
  %1 = shape.const_size 2
  %2 = shape.const_size 2
  return
}

func @test_to_extent_tensor(%arg: !shape.shape) -> tensor<3xindex> {
  %0 = shape.to_extent_tensor %arg : !shape.shape -> tensor<3xindex>
  return %0 : tensor<3xindex>
}

func @test_from_extent_tensor(%arg: tensor<?xindex>) -> !shape.shape {
  %0 = shape.from_extent_tensor %arg : tensor<?xindex>
  return %0 : !shape.shape
}

func @rank(%shape : !shape.shape) -> !shape.size {
  %rank = shape.rank %shape : !shape.shape -> !shape.size
  return %rank : !shape.size
}

func @rank_on_extent_tensor(%shape : tensor<?xindex>) -> index {
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

func @shape_eq_on_shapes(%a : !shape.shape, %b : !shape.shape) -> i1 {
  %result = shape.shape_eq %a, %b : !shape.shape, !shape.shape
  return %result : i1
}

func @shape_eq_on_tensors(%a : tensor<?xindex>, %b : tensor<?xindex>) -> i1 {
  %result = shape.shape_eq %a, %b : tensor<?xindex>, tensor<?xindex>
  return %result : i1
}

func @shape_eq_on_mixed(%a : tensor<?xindex>, %b : !shape.shape) -> i1 {
  %result = shape.shape_eq %a, %b : tensor<?xindex>, !shape.shape
  return %result : i1
}

func @get_extent_on_shape(%arg : !shape.shape) -> !shape.size {
  %c0 = shape.const_size 0
  %result = shape.get_extent %arg, %c0 :
      !shape.shape, !shape.size -> !shape.size
  return %result : !shape.size
}

func @get_extent_on_extent_tensor(%arg : tensor<?xindex>) -> index {
  %c0 = constant 0 : index
  %result = shape.get_extent %arg, %c0 : tensor<?xindex>, index -> index
  return %result : index
}

func @get_extent_on_mixed_operands(%arg : tensor<?xindex>) -> !shape.size {
  %c0 = shape.const_size 0
  %result = shape.get_extent %arg, %c0 : tensor<?xindex>, !shape.size -> !shape.size
  return %result : !shape.size
}

func @any() {
  %0 = shape.const_shape [1, 2, 3] : !shape.shape
  %1 = shape.const_shape [4, 5, 6] : !shape.shape
  %2 = "shape.any"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = shape.const_shape [1, 2, 3] : tensor<?xindex>
  %4 = shape.const_shape [4, 5, 6] : tensor<?xindex>
  %5 = "shape.any"(%3, %4) : (tensor<?xindex>, tensor<?xindex>) -> tensor<?xindex>
  return
}

func @num_elements_extent_tensor(%arg : tensor<?xindex>) -> index {
  %result = shape.num_elements %arg : tensor<?xindex> -> index
  return %result : index
}

func @num_elements_shape(%arg : !shape.shape) -> !shape.size {
  %result = shape.num_elements %arg : !shape.shape -> !shape.size
  return %result : !shape.size
}

// Testing invoking shape function from another. shape_equal_shapes is merely
// a trivial helper function to invoke elsewhere.
func @shape_equal_shapes(%a : !shape.value_shape, %b : !shape.value_shape) -> !shape.shape {
  %0 = shape.shape_of %a : !shape.value_shape -> !shape.shape
  %1 = shape.shape_of %b : !shape.value_shape -> !shape.shape
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  return %2 : !shape.shape
}
func @shape_with_shape(%a : !shape.value_shape, %b : !shape.value_shape) -> !shape.shape {
  %0 = shape.shape_of %a : !shape.value_shape -> !shape.shape
  %1 = shape.with_shape %b, %0 : !shape.value_shape, !shape.shape
  %2 = call @shape_equal_shapes(%a, %1) : (!shape.value_shape, !shape.value_shape) -> !shape.shape
  return %2 : !shape.shape
}

func @any_on_shape(%a : !shape.shape, %b : !shape.shape, %c : !shape.shape)
    -> !shape.shape {
  %result = shape.any %a, %b, %c
      : !shape.shape, !shape.shape, !shape.shape -> !shape.shape
  return %result : !shape.shape
}

func @any_on_mixed(%a : tensor<?xindex>,
                   %b : tensor<?xindex>,
                   %c : !shape.shape) -> !shape.shape {
  %result = shape.any %a, %b, %c
      : tensor<?xindex>, tensor<?xindex>, !shape.shape -> !shape.shape
  return %result : !shape.shape
}

func @any_on_extent_tensors(%a : tensor<?xindex>,
                            %b : tensor<?xindex>,
                            %c : tensor<?xindex>) -> tensor<?xindex> {
  %result = shape.any %a, %b, %c
      : tensor<?xindex>, tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return %result : tensor<?xindex>
}

func @is_broadcastable_on_extent_tensors(%a : tensor<?xindex>,
                                         %b : tensor<?xindex>) -> i1 {
  %result = shape.is_broadcastable %a, %b
      : tensor<?xindex>, tensor<?xindex>
  return %result : i1
}

func @is_broadcastable_on_shapes(%a : !shape.shape,
                                 %b : !shape.shape) -> i1 {
  %result = shape.is_broadcastable %a, %b
      : !shape.shape, !shape.shape
  return %result : i1
}

func @shape_upper_bounded_by_constant(%a: !shape.shape) -> !shape.shape {
  %0 = shape.const_shape [4, 57, 92] : !shape.shape
  %1 = shape.max %a, %0 : !shape.shape, !shape.shape -> !shape.shape
  %2 = shape.join %0, %1, error="exceeded element-wise upper bound" :
    !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}

func @shape_lower_bounded_by_constant(%a: !shape.shape) -> !shape.shape {
  %0 = shape.const_shape [4, 57, 92] : !shape.shape
  %1 = shape.min %a, %0 : !shape.shape, !shape.shape -> !shape.shape
  %2 = shape.join %0, %1, error="lower bound element-wise exceeded" :
    !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}

func @size_upper_bounded_by_constant(%a: !shape.size) -> !shape.size {
  %0 = shape.const_size 5
  %1 = shape.max %a, %0 : !shape.size, !shape.size -> !shape.size
  %2 = shape.join %0, %1, error="exceeded element-wise upper bound" :
    !shape.size, !shape.size -> !shape.size
  return %2 : !shape.size
}

func @size_lower_bounded_by_constant(%a: !shape.size) -> !shape.size {
  %0 = shape.const_size 9
  %1 = shape.min %a, %0 : !shape.size, !shape.size -> !shape.size
  %2 = shape.join %0, %1, error="lower bound element-wise exceeded" :
    !shape.size, !shape.size -> !shape.size
  return %2 : !shape.size
}
