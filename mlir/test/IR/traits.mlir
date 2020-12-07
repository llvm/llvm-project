// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: succeededSameOperandsElementType
func @succeededSameOperandsElementType(%t10x10 : tensor<10x10xf32>, %t1f: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>, %sf: f32) {
  "test.same_operand_element_type"(%t1f, %t1f) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi32>
  "test.same_operand_element_type"(%t1f, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> tensor<1xi32>
  "test.same_operand_element_type"(%t10x10, %v1) : (tensor<10x10xf32>, vector<1xf32>) -> tensor<1xi32>
  "test.same_operand_element_type"(%v1, %t1f) : (vector<1xf32>, tensor<1xf32>) -> tensor<1xi32>
  "test.same_operand_element_type"(%v1, %t1f) : (vector<1xf32>, tensor<1xf32>) -> tensor<121xi32>
  "test.same_operand_element_type"(%sf, %sf) : (f32, f32) -> i32
  "test.same_operand_element_type"(%sf, %t1f) : (f32, tensor<1xf32>) -> tensor<121xi32>
  "test.same_operand_element_type"(%sf, %v1) : (f32, vector<1xf32>) -> tensor<121xi32>
  "test.same_operand_element_type"(%sf, %t10x10) : (f32, tensor<10x10xf32>) -> tensor<121xi32>
  return
}

// -----

func @failedSameOperandElementType(%t1f: tensor<1xf32>, %t1i: tensor<1xi32>) {
  // expected-error@+1 {{requires the same element type for all operands}}
  "test.same_operand_element_type"(%t1f, %t1i) : (tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>
}

// -----

func @failedSameOperandAndResultElementType_no_operands() {
  // expected-error@+1 {{expected 2 operands, but found 0}}
  "test.same_operand_element_type"() : () -> tensor<1xf32>
}

// -----

func @failedSameOperandElementType_scalar_type_mismatch(%si: i32, %sf: f32) {
  // expected-error@+1 {{requires the same element type for all operands}}
  "test.same_operand_element_type"(%sf, %si) : (f32, i32) -> tensor<1xf32>
}

// -----

// CHECK: succeededSameOperandAndResultElementType
func @succeededSameOperandAndResultElementType(%t10x10 : tensor<10x10xf32>, %t1f: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>, %sf: f32) {
  "test.same_operand_and_result_element_type"(%t1f, %t1f) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_element_type"(%t1f, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_element_type"(%t10x10, %v1) : (tensor<10x10xf32>, vector<1xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_element_type"(%v1, %t1f) : (vector<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_element_type"(%v1, %t1f) : (vector<1xf32>, tensor<1xf32>) -> tensor<121xf32>
  "test.same_operand_and_result_element_type"(%sf, %sf) : (f32, f32) -> f32
  "test.same_operand_and_result_element_type"(%sf, %t1f) : (f32, tensor<1xf32>) -> tensor<121xf32>
  "test.same_operand_and_result_element_type"(%sf, %v1) : (f32, vector<1xf32>) -> tensor<121xf32>
  "test.same_operand_and_result_element_type"(%sf, %t10x10) : (f32, tensor<10x10xf32>) -> tensor<121xf32>
  return
}

// -----

func @failedSameOperandAndResultElementType_operand_result_mismatch(%t1f: tensor<1xf32>) {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  "test.same_operand_and_result_element_type"(%t1f, %t1f) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi32>
}

// -----

func @failedSameOperandAndResultElementType_operand_mismatch(%t1f: tensor<1xf32>, %t1i: tensor<1xi32>) {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  "test.same_operand_and_result_element_type"(%t1f, %t1i) : (tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>
}

// -----

func @failedSameOperandAndResultElementType_result_mismatch(%t1f: tensor<1xf32>) {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0:2 = "test.same_operand_and_result_element_type"(%t1f) : (tensor<1xf32>) -> (tensor<1xf32>, tensor<1xi32>)
}

// -----

func @failedSameOperandAndResultElementType_no_operands() {
  // expected-error@+1 {{expected 1 or more operands}}
  "test.same_operand_and_result_element_type"() : () -> tensor<1xf32>
}

// -----

func @failedSameOperandAndResultElementType_no_results(%t1f: tensor<1xf32>) {
  // expected-error@+1 {{expected 1 or more results}}
  "test.same_operand_and_result_element_type"(%t1f) : (tensor<1xf32>) -> ()
}

// -----

// CHECK: succeededSameOperandShape
func @succeededSameOperandShape(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %m10x10 : memref<10x10xi32>, %tr: tensor<*xf32>) {
  "test.same_operand_shape"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> ()
  "test.same_operand_shape"(%t10x10, %t10x10) : (tensor<10x10xf32>, tensor<10x10xf32>) -> ()
  "test.same_operand_shape"(%t1, %tr) : (tensor<1xf32>, tensor<*xf32>) -> ()
  "test.same_operand_shape"(%t10x10, %m10x10) : (tensor<10x10xf32>, memref<10x10xi32>) -> ()
  return
}

// -----

func @failedSameOperandShape_operand_mismatch(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>) {
  // expected-error@+1 {{requires the same shape for all operands}}
  "test.same_operand_shape"(%t1, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> ()
}

// -----

func @failedSameOperandShape_no_operands() {
  // expected-error@+1 {{expected 1 or more operands}}
  "test.same_operand_shape"() : () -> ()
}

// -----

// CHECK: succeededSameOperandAndResultShape
func @succeededSameOperandAndResultShape(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %tr: tensor<*xf32>, %t1d: tensor<?xf32>) {
  "test.same_operand_and_result_shape"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_shape"(%t10x10, %t10x10) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "test.same_operand_and_result_shape"(%t1, %tr) : (tensor<1xf32>, tensor<*xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_shape"(%t1, %t1d) : (tensor<1xf32>, tensor<?xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_shape"(%t1, %t1d) : (tensor<1xf32>, tensor<?xf32>) -> memref<1xf32>

  return
}

// -----

func @failedSameOperandAndResultShape_operand_result_mismatch(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>) {
  // expected-error@+1 {{requires the same shape for all operands and results}}
  "test.same_operand_and_result_shape"(%t1, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func @failedSameOperandAndResultShape_no_operands() {
  // expected-error@+1 {{expected 1 or more operands}}
  "test.same_operand_and_result_shape"() : () -> (tensor<1xf32>)
}

// -----

func @failedSameOperandAndResultShape_no_operands(%t1: tensor<1xf32>) {
  // expected-error@+1 {{expected 1 or more results}}
  "test.same_operand_and_result_shape"(%t1) : (tensor<1xf32>) -> ()
}

// -----

// CHECK: succeededSameOperandAndResultType
func @succeededSameOperandAndResultType(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %tr: tensor<*xf32>, %t1d: tensor<?xf32>, %i32 : i32) {
  "test.same_operand_and_result_type"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_type"(%t10x10, %t10x10) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "test.same_operand_and_result_type"(%t1, %tr) : (tensor<1xf32>, tensor<*xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_type"(%t1, %t1d) : (tensor<1xf32>, tensor<?xf32>) -> tensor<1xf32>
  "test.same_operand_and_result_type"(%i32, %i32) : (i32, i32) -> i32
  return
}

// -----

func @failedSameOperandAndResultType_operand_result_mismatch(%t10 : tensor<10xf32>, %t20 : tensor<20xf32>) {
  // expected-error@+1 {{requires the same type for all operands and results}}
  "test.same_operand_and_result_type"(%t10, %t20) : (tensor<10xf32>, tensor<20xf32>) -> tensor<10xf32>
}

// -----

func @failedElementwiseMappable_different_rankedness(%arg0: tensor<?xf32>, %arg1: tensor<*xf32>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type: found 'tensor<*xf32>' and 'tensor<?xf32>'}}
  %0 = "test.elementwise_mappable"(%arg0, %arg1) : (tensor<?xf32>, tensor<*xf32>) -> tensor<*xf32>
}

// -----

func @failedElementwiseMappable_different_rank(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type: found 'tensor<?x?xf32>' and 'tensor<?xf32>'}}
  %0 = "test.elementwise_mappable"(%arg0, %arg1) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
}

// -----

func @failedElementwiseMappable_different_shape(%arg0: tensor<?xf32>, %arg1: tensor<5xf32>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type: found 'tensor<5xf32>' and 'tensor<?xf32>'}}
  %0 = "test.elementwise_mappable"(%arg0, %arg1) : (tensor<?xf32>, tensor<5xf32>) -> tensor<?xf32>
}

// -----

func @failedElementwiseMappable_different_base_type(%arg0: vector<2xf32>, %arg1: tensor<2xf32>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type: found 'tensor<2xf32>' and 'vector<2xf32>'}}
  %0 = "test.elementwise_mappable"(%arg0, %arg1) : (vector<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return
}

// -----

func @failedElementwiseMappable_non_scalar_output(%arg0: vector<2xf32>) {
  // expected-error@+1 {{if an operand is non-scalar, then there must be at least one non-scalar result}}
  %0 = "test.elementwise_mappable"(%arg0) : (vector<2xf32>) -> f32
  return
}

// -----

func @failedElementwiseMappable_non_scalar_result_all_scalar_input(%arg0: f32) {
  // expected-error@+1 {{if a result is non-scalar, then at least one operand must be non-scalar}}
  %0 = "test.elementwise_mappable"(%arg0) : (f32) -> tensor<f32>
  return
}

// -----

func @failedElementwiseMappable_mixed_scalar_non_scalar_results(%arg0: tensor<10xf32>) {
  // expected-error@+1 {{if an operand is non-scalar, then all results must be non-scalar}}
  %0, %1 = "test.elementwise_mappable"(%arg0) : (tensor<10xf32>) -> (f32, tensor<10xf32>)
  return
}

// -----

func @failedElementwiseMappable_zero_results(%arg0: tensor<10xf32>) {
  // expected-error@+1 {{if an operand is non-scalar, then there must be at least one non-scalar result}}
  "test.elementwise_mappable"(%arg0) : (tensor<10xf32>) -> ()
  return
}

// -----

func @failedElementwiseMappable_zero_operands() {
  // expected-error@+1 {{if a result is non-scalar, then at least one operand must be non-scalar}}
  "test.elementwise_mappable"() : () -> (tensor<6xf32>)
  return
}

// -----

func @succeededElementwiseMappable(%arg0: vector<2xf32>) {
  // Check that varying element types are allowed.
  // CHECK: test.elementwise_mappable
  %0 = "test.elementwise_mappable"(%arg0) : (vector<2xf32>) -> vector<2xf16>
  return
}

// -----

func @failedHasParent_wrong_parent() {
  "some.op"() ({
   // expected-error@+1 {{'test.child' op expects parent op 'test.parent'}}
    "test.child"() : () -> ()
  }) : () -> ()
}

// -----

// CHECK: succeededParentOneOf
func @succeededParentOneOf() {
  "test.parent"() ({
    "test.child_with_parent_one_of"() : () -> ()
    "test.finish"() : () -> ()
   }) : () -> ()
  return
}

// -----

// CHECK: succeededParent1OneOf
func @succeededParent1OneOf() {
  "test.parent1"() ({
    "test.child_with_parent_one_of"() : () -> ()
    "test.finish"() : () -> ()
   }) : () -> ()
  return
}

// -----

func @failedParentOneOf_wrong_parent1() {
  "some.otherop"() ({
    // expected-error@+1 {{'test.child_with_parent_one_of' op expects parent op to be one of 'test.parent, test.parent1'}}
    "test.child_with_parent_one_of"() : () -> ()
    "test.finish"() : () -> ()
   }) : () -> ()
}


// -----

func @failedSingleBlockImplicitTerminator_empty_block() {
   // expected-error@+1 {{'test.SingleBlockImplicitTerminator' op expects a non-empty block}}
  "test.SingleBlockImplicitTerminator"() ({
  ^entry:
  }) : () -> ()
}

// -----

func @failedSingleBlockImplicitTerminator_too_many_blocks() {
   // expected-error@+1 {{'test.SingleBlockImplicitTerminator' op expects region #0 to have 0 or 1 block}}
  "test.SingleBlockImplicitTerminator"() ({
  ^entry:
    "test.finish" () : () -> ()
  ^other:
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

func @failedSingleBlockImplicitTerminator_missing_terminator() {
   // expected-error@+2 {{'test.SingleBlockImplicitTerminator' op expects regions to end with 'test.finish'}}
   // expected-note@+1 {{in custom textual format, the absence of terminator implies 'test.finish'}}
  "test.SingleBlockImplicitTerminator"() ({
  ^entry:
    "test.non_existent_op"() : () -> ()
  }) : () -> ()
}

// -----

// Test the invariants of operations with the Symbol Trait.

// expected-error@+1 {{requires string attribute 'sym_name'}}
"test.symbol"() {} : () -> ()

// -----

// expected-error@+1 {{requires visibility attribute 'sym_visibility' to be a string attribute}}
"test.symbol"() {sym_name = "foo_2", sym_visibility} : () -> ()

// -----

// expected-error@+1 {{visibility expected to be one of ["public", "private", "nested"]}}
"test.symbol"() {sym_name = "foo_2", sym_visibility = "foo"} : () -> ()

// -----

"test.symbol"() {sym_name = "foo_3", sym_visibility = "nested"} : () -> ()
"test.symbol"() {sym_name = "foo_4", sym_visibility = "private"} : () -> ()
"test.symbol"() {sym_name = "foo_5", sym_visibility = "public"} : () -> ()
"test.symbol"() {sym_name = "foo_6"} : () -> ()

// -----

// Test that operation with the SymbolTable Trait define a new symbol scope.
"test.symbol_scope"() ({
  func private @foo()
  "test.finish" () : () -> ()
}) : () -> ()
func private @foo() 

// -----

// Test that operation with the SymbolTable Trait fails with  too many blocks.
// expected-error@+1 {{Operations with a 'SymbolTable' must have exactly one block}}
"test.symbol_scope"() ({
  ^entry:
    "test.finish" () : () -> ()
  ^other:
    "test.finish" () : () -> ()
}) : () -> ()

// -----

func @failedMissingOperandSizeAttr(%arg: i32) {
  // expected-error @+1 {{requires 1D vector attribute 'operand_segment_sizes'}}
  "test.attr_sized_operands"(%arg, %arg, %arg, %arg) : (i32, i32, i32, i32) -> ()
}

// -----

func @failedOperandSizeAttrWrongType(%arg: i32) {
  // expected-error @+1 {{requires 1D vector attribute 'operand_segment_sizes'}}
  "test.attr_sized_operands"(%arg, %arg, %arg, %arg) {operand_segment_sizes = dense<[1, 1, 1, 1]>: tensor<4xi32>} : (i32, i32, i32, i32) -> ()
}

// -----

func @failedOperandSizeAttrWrongRank(%arg: i32) {
  // expected-error @+1 {{requires 1D vector attribute 'operand_segment_sizes'}}
  "test.attr_sized_operands"(%arg, %arg, %arg, %arg) {operand_segment_sizes = dense<[[1, 1], [1, 1]]>: vector<2x2xi32>} : (i32, i32, i32, i32) -> ()
}

// -----

func @failedOperandSizeAttrNegativeValue(%arg: i32) {
  // expected-error @+1 {{'operand_segment_sizes' attribute cannot have negative elements}}
  "test.attr_sized_operands"(%arg, %arg, %arg, %arg) {operand_segment_sizes = dense<[1, 1, -1, 1]>: vector<4xi32>} : (i32, i32, i32, i32) -> ()
}

// -----

func @failedOperandSizeAttrWrongTotalSize(%arg: i32) {
  // expected-error @+1 {{operand count (4) does not match with the total size (3) specified in attribute 'operand_segment_sizes'}}
  "test.attr_sized_operands"(%arg, %arg, %arg, %arg) {operand_segment_sizes = dense<[0, 1, 1, 1]>: vector<4xi32>} : (i32, i32, i32, i32) -> ()
}

// -----

func @failedOperandSizeAttrWrongCount(%arg: i32) {
  // expected-error @+1 {{'operand_segment_sizes' attribute for specifying operand segments must have 4 elements}}
  "test.attr_sized_operands"(%arg, %arg, %arg, %arg) {operand_segment_sizes = dense<[2, 1, 1]>: vector<3xi32>} : (i32, i32, i32, i32) -> ()
}

// -----

func @succeededOperandSizeAttr(%arg: i32) {
  // CHECK: test.attr_sized_operands
  "test.attr_sized_operands"(%arg, %arg, %arg, %arg) {operand_segment_sizes = dense<[0, 2, 1, 1]>: vector<4xi32>} : (i32, i32, i32, i32) -> ()
  return
}

// -----

func @failedMissingResultSizeAttr() {
  // expected-error @+1 {{requires 1D vector attribute 'result_segment_sizes'}}
  %0:4 = "test.attr_sized_results"() : () -> (i32, i32, i32, i32)
}

// -----

func @failedResultSizeAttrWrongType() {
  // expected-error @+1 {{requires 1D vector attribute 'result_segment_sizes'}}
  %0:4 = "test.attr_sized_results"() {result_segment_sizes = dense<[1, 1, 1, 1]>: tensor<4xi32>} : () -> (i32, i32, i32, i32)
}

// -----

func @failedResultSizeAttrWrongRank() {
  // expected-error @+1 {{requires 1D vector attribute 'result_segment_sizes'}}
  %0:4 = "test.attr_sized_results"() {result_segment_sizes = dense<[[1, 1], [1, 1]]>: vector<2x2xi32>} : () -> (i32, i32, i32, i32)
}

// -----

func @failedResultSizeAttrNegativeValue() {
  // expected-error @+1 {{'result_segment_sizes' attribute cannot have negative elements}}
  %0:4 = "test.attr_sized_results"() {result_segment_sizes = dense<[1, 1, -1, 1]>: vector<4xi32>} : () -> (i32, i32, i32, i32)
}

// -----

func @failedResultSizeAttrWrongTotalSize() {
  // expected-error @+1 {{result count (4) does not match with the total size (3) specified in attribute 'result_segment_sizes'}}
  %0:4 = "test.attr_sized_results"() {result_segment_sizes = dense<[0, 1, 1, 1]>: vector<4xi32>} : () -> (i32, i32, i32, i32)
}

// -----

func @failedResultSizeAttrWrongCount() {
  // expected-error @+1 {{'result_segment_sizes' attribute for specifying result segments must have 4 elements}}
  %0:4 = "test.attr_sized_results"() {result_segment_sizes = dense<[2, 1, 1]>: vector<3xi32>} : () -> (i32, i32, i32, i32)
}

// -----

func @succeededResultSizeAttr() {
  // CHECK: test.attr_sized_results
  %0:4 = "test.attr_sized_results"() {result_segment_sizes = dense<[0, 2, 1, 1]>: vector<4xi32>} : () -> (i32, i32, i32, i32)
  return
}

// -----

func @failedHasDominanceScopeOutsideDominanceFreeScope() -> () {
  "test.ssacfg_region"() ({
    test.graph_region {
      // expected-error @+1 {{operand #0 does not dominate this use}}
      %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
    }
    // expected-note @+1 {{operand defined here}}
    %1 = "baz"() : () -> (i64)
  }) : () -> ()
  return
}

// -----

// Ensure that SSACFG regions of operations in GRAPH regions are
// checked for dominance
func @illegalInsideDominanceFreeScope() -> () {
  test.graph_region {
    func @test() -> i1 {
    ^bb1:
      // expected-error @+1 {{operand #0 does not dominate this use}}
      %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
      // expected-note @+1 {{operand defined here}}
	   %1 = "baz"(%2#0) : (i1) -> (i64)
      return %2#1 : i1
    }
    "terminator"() : () -> ()
  }
  return
}

// -----

// Ensure that SSACFG regions of operations in GRAPH regions are
// checked for dominance
func @illegalCDFGInsideDominanceFreeScope() -> () {
  test.graph_region {
    func @test() -> i1 {
    ^bb1:
      // expected-error @+1 {{operand #0 does not dominate this use}}
      %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
      br ^bb4
    ^bb2:
      br ^bb2
    ^bb4:
      %1 = "foo"() : ()->i64   // expected-note {{operand defined here}}
		return %2#1 : i1
    }
     "terminator"() : () -> ()
  }
  return
}

// -----

// Ensure that GRAPH regions still have all values defined somewhere.
func @illegalCDFGInsideDominanceFreeScope() -> () {
  test.graph_region {
    // expected-error @+1 {{use of undeclared SSA value name}}
    %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
    "terminator"() : () -> ()
  }
  return
}

// -----

func @graph_region_cant_have_blocks() {
  test.graph_region {
    // expected-error@-1 {{'test.graph_region' op expects graph region #0 to have 0 or 1 blocks}}
  ^bb42:
    br ^bb43
  ^bb43:
    "terminator"() : () -> ()
  }
}
