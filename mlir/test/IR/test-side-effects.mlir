// RUN: mlir-opt %s -test-side-effects -verify-diagnostics

func.func @side_effect(%arg : index) {
  // expected-remark@+1 {{operation has no memory effects}}
  %0 = "test.side_effect_op"() {} : () -> i32
  
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'free' on resource '<Default>'}}
  %1 = "test.side_effect_op"() {effects = [
    {effect="read"}, {effect="free"}
  ]} : () -> i32
  
  // expected-remark@+1 {{found an instance of 'write' on resource '<Test>'}}
  %2 = "test.side_effect_op"() {effects = [
    {effect="write", test_resource}
  ]} : () -> i32
  
  // expected-remark@+1 {{found an instance of 'allocate' on a op result, on resource '<Test>'}}
  %3 = "test.side_effect_op"() {effects = [
    {effect="allocate", on_result, test_resource}
  ]} : () -> i32
  
  // expected-remark@+1 {{found an instance of 'read' on a symbol '@foo_ref', on resource '<Test>'}}
  "test.side_effect_op"() {effects = [
    {effect="read", on_reference = @foo_ref, test_resource}
  ]} : () -> i32
  
  // No _memory_ effects, but a parametric test effect.
  // expected-remark@+2 {{operation has no memory effects}}
  // expected-remark@+1 {{found a parametric effect with affine_map<(d0, d1) -> (d1, d0)>}}
  %4 = "test.side_effect_op"() {
    effect_parameter = affine_map<(i, j) -> (j, i)>
  } : () -> i32
  
  // Check with this unregistered operation to test the fallback on the dialect.
  // expected-remark@+1 {{found a parametric effect with affine_map<(d0, d1) -> (d1, d0)>}}
  %5 = "test.unregistered_side_effect_op"() {
    effect_parameter = affine_map<(i, j) -> (j, i)>
  } : () -> i32

  // expected-remark@+1 {{found an instance of 'allocate' on a op operand, on resource '<Test>'}}
  %6 = test.side_effect_with_region_op (%arg) {
  ^bb0(%arg0 : index):
    test.region_yield %arg0 : index 
  } {effects = [ {effect="allocate", on_operand, test_resource} ]} : index -> index

  // expected-remark@+1 {{found an instance of 'allocate' on a op result, on resource '<Test>'}}
  %7 = test.side_effect_with_region_op (%arg) {
  ^bb0(%arg0 : index):
    test.region_yield %arg0 : index 
  } {effects = [ {effect="allocate", on_result, test_resource} ]} : index -> index

  // expected-remark@+1 {{found an instance of 'allocate' on a block argument, on resource '<Test>'}}
  %8 = test.side_effect_with_region_op (%arg) {
  ^bb0(%arg0 : index):
    test.region_yield %arg0 : index 
  } {effects = [ {effect="allocate", on_argument, test_resource} ]} : index -> index

  func.return 
}
