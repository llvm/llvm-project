// RUN: mlir-opt %s -test-side-effects -verify-diagnostics

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

// expected-remark@+1 {{found an instance of 'allocate' on a value, on resource '<Test>'}}
%3 = "test.side_effect_op"() {effects = [
  {effect="allocate", on_result, test_resource}
]} : () -> i32
