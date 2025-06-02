// RUN: mlir-opt -verify-diagnostics -split-input-file %s

func.func @test_invalid_enum_case() -> () {
  // expected-error@+2 {{expected test::TestEnum to be one of: first, second, third}}
  // expected-error@+1 {{failed to parse TestEnumAttr}}
  test.op_with_enum #test<enum fourth>
}

// -----

func.func @test_invalid_enum_case() -> () {
  // expected-error@+1 {{expected test::TestEnum to be one of: first, second, third}}
  test.op_with_enum fourth
  // expected-error@+1 {{failed to parse TestEnumAttr}}
}

// -----

func.func @test_invalid_attr() -> () {
  // expected-error@+1 {{op attribute 'value' failed to satisfy constraint: a test enum}}
  "test.op_with_enum"() {value = 1 : index} : () -> ()
}

// -----

func.func @test_parse_invalid_attr() -> () {
  // expected-error@+2 {{expected valid keyword}}
  // expected-error@+1 {{failed to parse TestEnumAttr parameter 'value'}}
  test.op_with_enum 1 : index
}

// -----

func.func @test_non_keyword_prop_enum() -> () {
  // expected-error@+2 {{expected keyword for a test enum}}
  // expected-error@+1 {{invalid value for property value, expected a test enum}}
  test.op_with_enum_prop 0
  return
}

// -----

func.func @test_wrong_keyword_prop_enum() -> () {
  // expected-error@+2 {{expected one of [first, second, third] for a test enum, got: fourth}}
  // expected-error@+1 {{invalid value for property value, expected a test enum}}
  test.op_with_enum_prop fourth
}

// -----

func.func @test_bad_integer() -> () {
  // expected-error@+1 {{op property 'value' failed to satisfy constraint: a test enum}}
  "test.op_with_enum_prop"() <{value = 4 : i32}> {} : () -> ()
}

// -----

func.func @test_bit_enum_prop_not_keyword() -> () {
  // expected-error@+2 {{expected keyword for a test bit enum}}
  // expected-error@+1 {{invalid value for property value1, expected a test bit enum}}
  test.op_with_bit_enum_prop 0
  return
}

// -----

func.func @test_bit_enum_prop_wrong_keyword() -> () {
  // expected-error@+2 {{expected one of [read, write, execute] for a test bit enum, got: chroot}}
  // expected-error@+1 {{invalid value for property value1, expected a test bit enum}}
  test.op_with_bit_enum_prop read, chroot : ()
  return
}

// -----

func.func @test_bit_enum_prop_bad_value() -> () {
  // expected-error@+1 {{op property 'value2' failed to satisfy constraint: a test bit enum}}
  "test.op_with_bit_enum_prop"() <{value1 = 7 : i32, value2 = 8 : i32}> {} : () -> ()
  return
}

// -----

func.func @test_bit_enum_prop_named_wrong_keyword() -> () {
  // expected-error@+2 {{expected 'bit_enum'}}
  // expected-error@+1 {{invalid value for property value1, expected a test bit enum}}
  test.op_with_bit_enum_prop_named foo<read, execute>
  return
}

// -----

func.func @test_bit_enum_prop_named_not_open() -> () {
  // expected-error@+2 {{expected '<'}}
  // expected-error@+1 {{invalid value for property value1, expected a test bit enum}}
  test.op_with_bit_enum_prop_named bit_enum read, execute>
}

// -----

func.func @test_bit_enum_prop_named_not_closed() -> () {
  // expected-error@+2 {{expected '>'}}
  // expected-error@+1 {{invalid value for property value1, expected a test bit enum}}
  test.op_with_bit_enum_prop_named bit_enum<read, execute +
}
