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

// `operandSegmentSizes` is inferable from the individually-spelled-out
// variadic groups in this op's format, so the custom parser never reads this
// key at all: presence of the key is rejected as an unrecognized key, just
// like any other name the op doesn't know about.
func.func @unknown_optional_operand_segment_sizes(%arg0: i64) {
  // expected-error@+1 {{unknown key '"operandSegmentSizes"' when parsing properties dictionary}}
  test.variadic_segment_prop %arg0, %arg0 : %arg0 : i64, i64 : i64 <{operandSegmentSizes = "bad"}> end
  return
}

// -----

func.func @unknown_optional_result_segment_sizes(%arg0: i64) {
  // expected-error@+1 {{unknown key '"resultSegmentSizes"' when parsing properties dictionary}}
  test.variadic_segment_prop %arg0, %arg0 : %arg0 : i64, i64 : i64 <{resultSegmentSizes = "bad"}> end
  return
}

// -----

func.func @valid_all_properties() {
  "test.with_properties"() <{a = 32 : i64, array = array<i64: 1, 2, 3, 4>, array32 = array<i32: 5, 6>, b = "foo", c = "bar", flag = true}> : () -> ()
  return
}
