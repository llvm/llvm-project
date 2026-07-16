// RUN: mlir-opt -verify-diagnostics -split-input-file %s


// -----

func.func @wrong_string_prop_type() {
  // expected-error@+1 {{for `c`: expected StringAttr}}
  "test.with_properties"() <{b = "foo", c = 32 : i64}> : () -> ()
  return
}

// -----

func.func @wrong_bool_prop_type() {
  // expected-error@+1 {{for `flag`: expected BoolAttr}}
  "test.with_properties"() <{b = "foo", flag = "bar"}> : () -> ()
  return
}

// -----

func.func @wrong_integer_prop_type() {
  // expected-error@+1 {{for `a`: expected IntegerAttr}}
  "test.with_properties"() <{b = "foo", a = "bar"}> : () -> ()
  return
}

// -----

func.func @wrong_dense_i64_array_prop_type() {
  // expected-error@+1 {{for `array`: expected DenseI64ArrayAttr}}
  "test.with_properties"() <{b = "foo", array = array<i32: 1, 2, 3, 4>}> : () -> ()
  return
}

// -----

func.func @wrong_dense_i32_array_prop_type() {
  // expected-error@+1 {{for `array32`: expected DenseI32ArrayAttr}}
  "test.with_properties"() <{b = "foo", array32 = array<i64: 5, 6>}> : () -> ()
  return
}

// -----

func.func @valid_all_properties() {
  "test.with_properties"() <{a = 32 : i64, array = array<i64: 1, 2, 3, 4>, array32 = array<i32: 5, 6>, b = "foo", c = "bar", flag = true}> : () -> ()
  return
}
