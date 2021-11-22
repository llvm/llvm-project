// RUN: mlir-opt --split-input-file %s --verify-diagnostics

func private @test_ugly_attr_cannot_be_pretty() -> () attributes {
  // expected-error@+1 {{expected 'begin'}}
  attr = #test.attr_ugly
}

// -----

func private @test_ugly_attr_no_mnemonic() -> () attributes {
  // expected-error@+1 {{expected valid keyword}}
  attr = #test<"">
}

// -----

func private @test_ugly_attr_parser_dispatch() -> () attributes {
  // expected-error@+1 {{expected 'begin'}}
  attr = #test<"attr_ugly">
}

// -----

func private @test_ugly_attr_missing_parameter() -> () attributes {
  // expected-error@+2 {{failed to parse TestAttrUgly parameter 'attr'}}
  // expected-error@+1 {{expected non-function type}}
  attr = #test<"attr_ugly begin">
}

// -----

func private @test_ugly_attr_missing_literal() -> () attributes {
  // expected-error@+1 {{expected 'end'}}
  attr = #test<"attr_ugly begin \"string_attr\"">
}

// -----

func private @test_pretty_attr_expects_less() -> () attributes {
  // expected-error@+1 {{expected '<'}}
  attr = #test.attr_with_format
}

// -----

func private @test_pretty_attr_missing_param() -> () attributes {
  // expected-error@+2 {{expected integer value}}
  // expected-error@+1 {{failed to parse TestAttrWithFormat parameter 'one'}}
  attr = #test.attr_with_format<>
}

// -----

func private @test_parse_invalid_param() -> () attributes {
  // Test parameter parser failure is propagated
  // expected-error@+2 {{expected integer value}}
  // expected-error@+1 {{failed to parse TestAttrWithFormat parameter 'one'}}
  attr = #test.attr_with_format<"hi">
}

// -----

func private @test_pretty_attr_invalid_syntax() -> () attributes {
  // expected-error@+1 {{expected ':'}}
  attr = #test.attr_with_format<42>
}

// -----

func private @test_struct_missing_key() -> () attributes {
  // expected-error@+2 {{expected valid keyword}}
  // expected-error@+1 {{expected a parameter name in struct}}
  attr = #test.attr_with_format<42 :>
}

// -----

func private @test_struct_unknown_key() -> () attributes {
  // expected-error@+1 {{duplicate or unknown struct parameter}}
  attr = #test.attr_with_format<42 : nine = "foo">
}

// -----

func private @test_struct_duplicate_key() -> () attributes {
  // expected-error@+1 {{duplicate or unknown struct parameter}}
  attr = #test.attr_with_format<42 : two = "foo", two = "bar">
}

// -----

func private @test_struct_not_enough_values() -> () attributes {
  // expected-error@+1 {{expected ','}}
  attr = #test.attr_with_format<42 : two = "foo">
}

// -----

func private @test_parse_param_after_struct() -> () attributes {
  // expected-error@+2 {{expected non-function type}}
  // expected-error@+1 {{failed to parse TestAttrWithFormat parameter 'three'}}
  attr = #test.attr_with_format<42 : two = "foo", four = [1, 2, 3] : >
}

// -----

// expected-error@+1 {{expected '<'}}
func private @test_invalid_type() -> !test.type_with_format

// -----

// expected-error@+2 {{expected integer value}}
// expected-error@+1 {{failed to parse TestTypeWithFormat parameter 'one'}}
func private @test_pretty_type_invalid_param() -> !test.type_with_format<>

// -----

// expected-error@+2 {{expected ':'}}
// expected-error@+1 {{failed to parse TestTypeWithFormat parameter 'three'}}
func private @test_type_syntax_error() -> !test.type_with_format<42, two = "hi", three = #test.attr_with_format<42>>

// -----

func private @test_verifier_fails() -> () attributes {
  // expected-error@+1 {{expected 'one' to equal 'four.size()'}}
  attr = #test.attr_with_format<42 : two = "hello", four = [1, 2, 3] : 42 : i64>
}

// -----

func private @test_attr_with_type_failed_to_parse_type() -> () attributes {
  // expected-error@+2 {{invalid kind of type specified}}
  // expected-error@+1 {{failed to parse TestAttrWithTypeParam parameter 'int_type'}}
  attr = #test.attr_with_type<vector<4xi32>, vector<4xi32>>
}
