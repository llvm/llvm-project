// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @below {{properties dictionary is missing required property: prop}}
test.with_custom_prop_dict <attr = 1>

// -----

// expected-error @below {{properties dictionary is missing required attribute: attr}}
test.with_custom_prop_dict <prop = 2>

// -----

// expected-error @below {{duplicate or unknown property in properties dictionary: prop}}
test.with_custom_prop_dict <attr = 1, prop = 2, prop = 3>

// -----

// expected-error @below {{duplicate or unknown property in properties dictionary: unknown}}
test.with_custom_prop_dict <attr = 1, prop = 2, unknown = 3>

// -----

// A required property dictionary cannot be omitted entirely.
// expected-error @below {{properties dictionary is missing required property: prop}}
test.with_custom_prop_dict

// -----

// expected-error @below {{expected integer value}}
test.with_custom_prop_dict <attr = 1, prop = bad>

// -----

// expected-error @below {{invalid value for property prop}}
test.with_wrapped_properties <prop = 1 : i64>

// -----

// A required keyed value may not succeed without consuming a token.
// expected-error @below {{expected attribute value}}
test.with_key_value_parser_boundaries <values = array<i64: 1>, maybe = >

// -----

%c0 = arith.constant 0 : i64
// A segment-size property inferred later in the parser must not be accepted
// and then silently overwritten.
// expected-error @below {{unknown property in properties dictionary: operandSegmentSizes}}
test.variadic_segment_prop %c0 : %c0 : i64 : i64 <operandSegmentSizes = [1, 1]> end

// -----

%c0 = arith.constant 0 : i64
// expected-error @below {{unknown property in properties dictionary: resultSegmentSizes}}
test.variadic_segment_prop %c0 : %c0 : i64 : i64 <resultSegmentSizes = [1, 1]> end

// -----

%c0 = arith.constant 0 : i64
// expected-error @below {{properties dictionary is missing required property: operandSegmentSizes}}
test.variadic_segment_prop_bulk_type(%c0, %c0, %c0) : (i64, i64, i64) -> (i64, i64, i64) <resultSegmentSizes = [2, 1]>

// -----

%c0 = arith.constant 0 : i64
// expected-error @below {{properties dictionary is missing required property: resultSegmentSizes}}
test.variadic_segment_prop_bulk_type(%c0, %c0, %c0) : (i64, i64, i64) -> (i64, i64, i64) <operandSegmentSizes = [2, 1]>

// -----

%c0 = arith.constant 0 : i64
// expected-error @below {{expected 2 entries for operandSegmentSizes}}
test.variadic_segment_prop_bulk_type(%c0, %c0, %c0) : (i64, i64, i64) -> (i64, i64, i64) <operandSegmentSizes = [3], resultSegmentSizes = [2, 1]>

// -----

%c0 = arith.constant 0 : i64
// expected-error @below {{expected 2 entries for resultSegmentSizes}}
test.variadic_segment_prop_bulk_type(%c0, %c0, %c0) : (i64, i64, i64) -> (i64, i64, i64) <operandSegmentSizes = [2, 1], resultSegmentSizes = [3]>
