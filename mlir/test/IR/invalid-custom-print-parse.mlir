// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error@+2 {{invalid dimension}}
// expected-error@+1 {{custom op 'test.custom_dimension_list_attr' Failed parsing dimension list.}}
test.custom_dimension_list_attr dimension_list = 1x-1

// -----

// expected-error@+1 {{custom op 'test.custom_dimension_list_attr' Failed parsing dimension list. Did you mean an empty list? It must be denoted by "[]".}}
test.custom_dimension_list_attr dimension_list = -1

// -----

// expected-error@+2 {{expected ']'}}
// expected-error@+1 {{custom op 'test.custom_dimension_list_attr' Failed parsing dimension list.}}
test.custom_dimension_list_attr dimension_list = [2x3]

// -----

// expected-error @below {{expected attribute value}}
test.optional_custom_attr foo
