// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @func_op() {
  // expected-error@+1 {{expected valid '@'-identifier for symbol name}}
  func missingsigil() -> (i1, index, f32)
  return
}

// -----

func @func_op() {
  // expected-error@+1 {{expected type instead of SSA identifier}}
  func @mixed_named_arguments(f32, %a : i32) {
    return
  }
  return
}

// -----

func @func_op() {
  // expected-error@+1 {{expected SSA identifier}}
  func @mixed_named_arguments(%a : i32, f32) -> () {
    return
  }
  return
}

// -----

func @func_op() {
  // expected-error@+1 {{entry block must have 1 arguments to match function signature}}
  func @mixed_named_arguments(f32) {
  ^entry:
    return
  }
  return
}

// -----

func @func_op() {
  // expected-error@+1 {{type of entry block argument #0('i32') must match the type of the corresponding argument in function signature('f32')}}
  func @mixed_named_arguments(f32) {
  ^entry(%arg : i32):
    return
  }
  return
}

// -----

// expected-error@+1 {{expected non-function type}}
func @f() -> (foo

// -----

// expected-error@+1 {{expected attribute name}}
func @f() -> (i1 {)

// -----

// expected-error@+1 {{invalid to use 'test.invalid_attr'}}
func @f(%arg0: i64 {test.invalid_attr}) {
  return
}

// -----

// expected-error@+1 {{invalid to use 'test.invalid_attr'}}
func @f(%arg0: i64) -> (i64 {test.invalid_attr}) {
  return %arg0 : i64
}

// -----

// expected-error@+1 {{symbol declaration cannot have public visibility}}
func @invalid_public_declaration()

// -----

// expected-error@+1 {{'sym_visibility' is an inferred attribute and should not be specified in the explicit attribute dictionary}}
func @legacy_visibility_syntax() attributes { sym_visibility = "private" }

// -----

// expected-error@+1 {{'sym_name' is an inferred attribute and should not be specified in the explicit attribute dictionary}}
func private @invalid_symbol_name_attr() attributes { sym_name = "x" }

// -----

// expected-error@+1 {{'type' is an inferred attribute and should not be specified in the explicit attribute dictionary}}
func private @invalid_symbol_type_attr() attributes { type = "x" }

// -----

// expected-error@+1 {{argument attribute array `arg_attrs` to have the same number of elements as the number of function arguments}}
func private @invalid_arg_attrs() attributes { arg_attrs = [{}] }

// -----

// expected-error@+1 {{expects argument attribute dictionary to be a DictionaryAttr, but got `10 : i64`}}
func private @invalid_arg_attrs(i32) attributes { arg_attrs = [10] }

// -----

// expected-error@+1 {{result attribute array `res_attrs` to have the same number of elements as the number of function results}}
func private @invalid_res_attrs() attributes { res_attrs = [{}] }

// -----

// expected-error@+1 {{expects result attribute dictionary to be a DictionaryAttr, but got `10 : i64`}}
func private @invalid_res_attrs() -> i32 attributes { res_attrs = [10] }
