// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @func_op() {
  // expected-error@+1 {{expected valid '@'-identifier for symbol name}}
  func.func missingsigil() -> (i1, index, f32)
  return
}

// -----

func.func @func_op() {
  // expected-error@+1 {{expected type instead of SSA identifier}}
  func.func @mixed_named_arguments(f32, %a : i32) {
    return
  }
  return
}

// -----

func.func @func_op() {
  // expected-error@+1 {{expected SSA identifier}}
  func.func @mixed_named_arguments(%a : i32, f32) -> () {
    return
  }
  return
}

// -----

func.func @func_op() {
  // expected-error@+1 {{op symbol's parent must have the SymbolTable trait}}
  func.func @mixed_named_arguments(f32) {
  ^entry:
    return
  }
  return
}

// -----

func.func @func_op() {
  // expected-error@+1 {{op symbol's parent must have the SymbolTable trait}}
  func.func @mixed_named_arguments(f32) {
  ^entry(%arg : i32):
    return
  }
  return
}

// -----

// expected-error@+1 {{expected non-function type}}
func.func @f() -> (foo

// -----

// expected-error@+1 {{expected attribute name}}
func.func @f() -> (i1 {)

// -----

// expected-error@+1 {{invalid to use 'test.invalid_attr'}}
func.func @f(%arg0: i64 {test.invalid_attr}) {
  return
}

// -----

// expected-error@+1 {{invalid to use 'test.invalid_attr'}}
func.func @f(%arg0: i64) -> (i64 {test.invalid_attr}) {
  return %arg0 : i64
}

// -----

// expected-error@+1 {{symbol declaration cannot have public visibility}}
func.func @invalid_public_declaration()

// -----

// expected-error@+1 {{'sym_visibility' is an inferred attribute and should not be specified in the explicit attribute dictionary}}
func.func @legacy_visibility_syntax() attributes { sym_visibility = "private" }

// -----

// expected-error@+1 {{'sym_name' is an inferred attribute and should not be specified in the explicit attribute dictionary}}
func.func private @invalid_symbol_name_attr() attributes { sym_name = "x" }

// -----

// expected-error@+1 {{'function_type' is an inferred attribute and should not be specified in the explicit attribute dictionary}}
func.func private @invalid_symbol_type_attr() attributes { function_type = "x" }

// -----

// expected-error@+1 {{argument attribute array to have the same number of elements as the number of function arguments}}
func.func private @invalid_arg_attrs() attributes { arg_attrs = [{}] }

// -----


// expected-error@+1 {{result attribute array to have the same number of elements as the number of function results}}
func.func private @invalid_res_attrs() attributes { res_attrs = [{}] }
