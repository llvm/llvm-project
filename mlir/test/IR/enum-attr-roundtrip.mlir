// RUN: mlir-opt %s | mlir-opt -test-greedy-patterns | FileCheck %s

// CHECK-LABEL: @test_enum_attr_roundtrip
func.func @test_enum_attr_roundtrip() -> () {
  // CHECK: value = #test<enum first>
  "test.op"() {value = #test<enum first>} : () -> ()
  // CHECK: value = #test<enum second>
  "test.op"() {value = #test<enum second>} : () -> ()
  // CHECK: value = #test<enum third>
  "test.op"() {value = #test<enum third>} : () -> ()
  return
}

// CHECK-LABEL: @test_op_with_enum
func.func @test_op_with_enum() -> () {
  // CHECK: test.op_with_enum third
  test.op_with_enum third
  return
}

// CHECK-LABEL: @test_match_op_with_enum
func.func @test_match_op_with_enum() -> () {
  // CHECK: test.op_with_enum third tag 0 : i32
  test.op_with_enum third tag 0 : i32
  // CHECK: test.op_with_enum second tag 1 : i32
  test.op_with_enum first tag 0 : i32
  return
}

// CHECK-LABEL: @test_match_op_with_bit_enum
func.func @test_match_op_with_bit_enum() -> () {
  // CHECK: test.op_with_bit_enum <write> tag 0 : i32
  test.op_with_bit_enum <write> tag 0 : i32
  // CHECK: test.op_with_bit_enum <read, execute> tag 1 : i32
  test.op_with_bit_enum <execute, write> tag 0 : i32
  return
}

// CHECK-LABEL: @test_enum_prop
func.func @test_enum_prop() -> () {
  // CHECK: test.op_with_enum_prop first
  test.op_with_enum_prop first

  // CHECK: test.op_with_enum_prop first
  "test.op_with_enum_prop"() <{value = 0 : i32}> {} : () -> ()

  // CHECK: test.op_with_enum_prop_attr_form <{value = 0 : i32}>
  test.op_with_enum_prop_attr_form <{value = 0 : i32}>
  // CHECK: test.op_with_enum_prop_attr_form <{value = 1 : i32}>
  test.op_with_enum_prop_attr_form <{value = #test<enum second>}>

  // CHECK: test.op_with_enum_prop_attr_form_always <{value = #test<enum first>}>
  test.op_with_enum_prop_attr_form_always <{value = #test<enum first>}>
  // CHECK: test.op_with_enum_prop_attr_form_always  <{value = #test<enum second>}
  test.op_with_enum_prop_attr_form_always <{value = #test<enum second>}>

  return
}

// CHECK-LABEL: @test_two_default_enum_props
func.func @test_two_default_enum_props() -> () {
  // Both default: both elided.
  // CHECK: test.op_with_two_default_enum_props{{$}}
  test.op_with_two_default_enum_props

  // value1 non-default, value2 default (dog): only value1 printed.
  // CHECK: test.op_with_two_default_enum_props second{{$}}
  test.op_with_two_default_enum_props second

  // value1 default (first), value2 non-default: only value2 printed.
  // Key test: value1 parser must not consume "cat" (not a TestEnum keyword).
  // CHECK: test.op_with_two_default_enum_props cat{{$}}
  test.op_with_two_default_enum_props cat

  // Both non-default.
  // CHECK: test.op_with_two_default_enum_props second cat{{$}}
  test.op_with_two_default_enum_props second cat

  return
}

// CHECK-LABEL: @test_two_default_bit_enum_props
func.func @test_two_default_bit_enum_props() -> () {
  // Both default: both elided.
  // CHECK: test.op_with_two_default_bit_enum_props{{$}}
  test.op_with_two_default_bit_enum_props

  // value1 non-default (write), value2 default (user): only value1 printed.
  // CHECK: test.op_with_two_default_bit_enum_props write{{$}}
  test.op_with_two_default_bit_enum_props write

  // value1 default (read), value2 non-default: only value2 printed.
  // Key test: value1 parser must not consume "group" (not a TestBitEnum keyword).
  // CHECK: test.op_with_two_default_bit_enum_props group{{$}}
  test.op_with_two_default_bit_enum_props group

  // Both non-default.
  // CHECK: test.op_with_two_default_bit_enum_props write group{{$}}
  test.op_with_two_default_bit_enum_props write group

  return
}

// CHECK-LABEL @test_bit_enum_prop()
func.func @test_bit_enum_prop() -> () {
  // CHECK: test.op_with_bit_enum_prop read : ()
  test.op_with_bit_enum_prop read read : ()

  // CHECK: test.op_with_bit_enum_prop read, write write, execute
  test.op_with_bit_enum_prop read, write write, execute : ()

  // CHECK: test.op_with_bit_enum_prop read, execute write
  "test.op_with_bit_enum_prop"() <{value1 = 5 : i32, value2 = 2 : i32}> {} : () -> ()

  // CHECK: test.op_with_bit_enum_prop read, write, execute
  test.op_with_bit_enum_prop read, write, execute : ()

  // CHECK: test.op_with_bit_enum_prop_named bit_enum<read>{{$}}
  test.op_with_bit_enum_prop_named bit_enum<read> bit_enum<read>
  // CHECK: test.op_with_bit_enum_prop_named bit_enum<read, write> bit_enum<write, execute>
  test.op_with_bit_enum_prop_named bit_enum<read, write> bit_enum<write, execute>
  // CHECK: test.op_with_bit_enum_prop_named bit_enum<read, write, execute>
  test.op_with_bit_enum_prop_named bit_enum<read, write, execute>

  return
}
