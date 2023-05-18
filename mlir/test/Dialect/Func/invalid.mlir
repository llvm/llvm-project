// RUN: mlir-opt -allow-unregistered-dialect -split-input-file %s -verify-diagnostics

func.func @unsupported_attribute() {
  // expected-error @+1 {{invalid kind of attribute specified}}
  %0 = constant "" : index
  return
}

// -----

func.func private @return_i32_f32() -> (i32, f32)

func.func @call() {
  // expected-error @+3 {{op result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f32', 'i32'}}
  // expected-note @+1 {{function result types: 'i32', 'f32'}}
  %0:2 = call @return_i32_f32() : () -> (f32, i32)
  return
}

// -----

func.func @resulterror() -> i32 {
^bb42:
  return    // expected-error {{'func.return' op has 0 operands, but enclosing function (@resulterror) returns 1}}
}

// -----

func.func @return_type_mismatch() -> i32 {
  %0 = "foo"() : ()->f32
  return %0 : f32  // expected-error {{type of return operand 0 ('f32') doesn't match function result type ('i32') in function @return_type_mismatch}}
}

// -----

func.func @return_inside_loop() {
  affine.for %i = 1 to 100 {
    // expected-error@+1 {{'func.return' op expects parent op 'func.func'}}
    func.return
  }
  return
}

// -----

// expected-error@+1 {{expected non-function type}}
func.func @func_variadic(...)

// -----

func.func @foo() {
^bb0:
  %x = constant @foo : (i32) -> ()  // expected-error {{reference to function with mismatched type}}
  return
}

// -----

func.func @undefined_function() {
^bb0:
  %x = constant @qux : (i32) -> ()  // expected-error {{reference to undefined function 'qux'}}
  return
}

// -----

#map1 = affine_map<(i)[j] -> (i+j)>

func.func @bound_symbol_mismatch(%N : index) {
  affine.for %i = #map1(%N) to 100 {
  // expected-error@-1 {{symbol operand count and affine map symbol count must match}}
  }
  return
}

// -----

#map1 = affine_map<(i)[j] -> (i+j)>

func.func @bound_dim_mismatch(%N : index) {
  affine.for %i = #map1(%N, %N)[%N] to 100 {
  // expected-error@-1 {{dim operand count and affine map dim count must match}}
  }
  return
}

// -----

func.func @large_bound() {
  affine.for %i = 1 to 9223372036854775810 {
  // expected-error@-1 {{integer constant out of range for attribute}}
  }
  return
}

// -----

func.func @max_in_upper_bound(%N : index) {
  affine.for %i = 1 to max affine_map<(i)->(N, 100)> { //expected-error {{expected attribute value}}
  }
  return
}

// -----

func.func @step_typo() {
  affine.for %i = 1 to 100 step -- 1 { //expected-error {{expected constant integer}}
  }
  return
}

// -----

func.func @invalid_bound_map(%N : i32) {
  affine.for %i = 1 to affine_map<(i)->(j)>(%N) { //expected-error {{use of undeclared identifier}}
  }
  return
}

// -----

// expected-error @+1 {{expected '(' in integer set constraint list}}
#set0 = affine_set<(i)[N, M] : )i >= 0)>

// -----
#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @invalid_if_operands1(%N : index) {
  affine.for %i = 1 to 10 {
    affine.if #set0(%i) {
    // expected-error@-1 {{symbol operand count and integer set symbol count must match}}

// -----
#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @invalid_if_operands2(%N : index) {
  affine.for %i = 1 to 10 {
    affine.if #set0()[%N] {
    // expected-error@-1 {{dim operand count and integer set dim count must match}}

// -----
#set0 = affine_set<(i)[N] : (i >= 0, N - i >= 0)>

func.func @invalid_if_operands3(%N : index) {
  affine.for %i = 1 to 10 {
    affine.if #set0(%i)[%i] {
    // expected-error@-1 {{operand cannot be used as a symbol}}
    }
  }
  return
}

// -----

func.func @redundant_signature(%a : i32) -> () {
^bb0(%b : i32):  // expected-error {{invalid block name in region with named arguments}}
  return
}

// -----

func.func @mixed_named_arguments(%a : i32,
                               f32) -> () {
    // expected-error @-1 {{expected SSA identifier}}
  return
}

// -----

func.func @mixed_named_arguments(f32,
                               %a : i32) -> () { // expected-error {{expected type instead of SSA identifier}}
  return
}

// -----

// expected-error @+1 {{@ identifier expected to start with letter or '_'}}
func.func @$invalid_function_name()

// -----

// expected-error @+1 {{arguments may only have dialect attributes}}
func.func private @invalid_func_arg_attr(i1 {non_dialect_attr = 10})

// -----

// expected-error @+1 {{results may only have dialect attributes}}
func.func private @invalid_func_result_attr() -> (i1 {non_dialect_attr = 10})

// -----

func.func @foo() {} // expected-error {{expected non-empty function body}}
