// RUN: mlir-opt -verify-diagnostics -split-input-file %s

func @test_invalid_enum_case() -> () {
  // expected-error@+2 {{expected test::TestEnum to be one of: first, second, third}}
  // expected-error@+1 {{failed to parse TestEnumAttr}}
  test.op_with_enum #test<"enum fourth">
}

// -----

func @test_invalid_enum_case() -> () {
  // expected-error@+1 {{expected test::TestEnum to be one of: first, second, third}}
  test.op_with_enum fourth
  // expected-error@+1 {{failed to parse TestEnumAttr}}
}

// -----

func @test_invalid_attr() -> () {
  // expected-error@+1 {{op attribute 'value' failed to satisfy constraint: a test enum}}
  "test.op_with_enum"() {value = 1 : index} : () -> ()
}

// -----

func @test_parse_invalid_attr() -> () {
  // expected-error@+2 {{expected valid keyword}}
  // expected-error@+1 {{failed to parse TestEnumAttr parameter 'value'}}
  test.op_with_enum 1 : index
}
