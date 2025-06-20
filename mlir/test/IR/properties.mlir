// # RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s
// # RUN: mlir-opt %s -mlir-print-op-generic -split-input-file  | mlir-opt -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// CHECK:   test.with_properties
// CHECK-SAME: a = 32, b = "foo", c = "bar", flag = true, array = [1, 2, 3, 4]{{$}}
// GENERIC:   "test.with_properties"()
// GENERIC-SAME: <{a = 32 : i64, array = array<i64: 1, 2, 3, 4>, b = "foo", c = "bar", flag = true}> : () -> ()
test.with_properties a = 32, b = "foo", c = "bar", flag = true, array = [1, 2, 3, 4]

// CHECK:   test.with_nice_properties
// CHECK-SAME:    "foo bar" is -3{{$}}
// GENERIC: "test.with_nice_properties"()
// GENERIC-SAME:  <{prop = {label = "foo bar", value = -3 : i32}}> : () -> ()
test.with_nice_properties "foo bar" is -3

// CHECK:   test.with_wrapped_properties
// CHECK-SAME:    <{prop = "content for properties"}>{{$}}
// GENERIC: "test.with_wrapped_properties"()
// GENERIC-SAME:  <{prop = "content for properties"}> : () -> ()
test.with_wrapped_properties <{prop = "content for properties"}>

// CHECK: test.empty_properties
// GENERIC: "test.empty_properties"()
test.empty_properties

// CHECK: test.using_property_in_custom
// CHECK-SAME: [1, 4, 20]{{$}}
// GENERIC: "test.using_property_in_custom"()
// GENERIC-SAME: prop = array<i64: 1, 4, 20>
test.using_property_in_custom [1, 4, 20]

// CHECK: test.using_property_ref_in_custom
// CHECK-SAME: 1 + 4 = 5{{$}}
// GENERIC: "test.using_property_ref_in_custom"()
// GENERIC-SAME: <{
// GENERIC-SAME: first = 1
// GENERIC-SAME: second = 4
// GENERIC-SAME: }>
test.using_property_ref_in_custom 1 + 4 = 5

// Tests that the variadic segment size properties are elided.
// CHECK: %[[CI64:.*]] = arith.constant
// CHECK-NEXT: test.variadic_segment_prop %[[CI64]], %[[CI64]] : %[[CI64]] : i64, i64 : i64 end
// GENERIC: %[[CI64:.*]] = "arith.constant"()
// GENERIC-NEXT: "test.variadic_segment_prop"(%[[CI64]], %[[CI64]], %[[CI64]]) <{operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 2, 1>}> : (i64, i64, i64) -> (i64, i64, i64)
%ci64 = arith.constant 0 : i64
test.variadic_segment_prop %ci64, %ci64 : %ci64 : i64, i64 : i64 end

// CHECK:   test.with_default_valued_properties na{{$}}
// GENERIC: "test.with_default_valued_properties"()
// GENERIC-SAME: <{a = 0 : i32, b = "", c = -1 : i32, unit = false}> : () -> ()
test.with_default_valued_properties 0 "" -1 unit_absent

// CHECK:   test.with_default_valued_properties 1 "foo" 0 unit{{$}}
// GENERIC: "test.with_default_valued_properties"()
// GENERIC-SAME: <{a = 1 : i32, b = "foo", c = 0 : i32, unit}> : () -> ()
test.with_default_valued_properties 1 "foo" 0 unit

// CHECK:   test.with_optional_properties
// CHECK-SAME: simple = 0
// GENERIC: "test.with_optional_properties"()
// GENERIC-SAME:  <{hasDefault = [], hasUnit = false, longSyntax = [], maybeUnit = [], nested = [], nonTrivialStorage = [], simple = [0]}> : () -> ()
test.with_optional_properties simple = 0

// CHECK:   test.with_optional_properties{{$}}
// GENERIC: "test.with_optional_properties"()
// GENERIC-SAME: simple = []
test.with_optional_properties

// CHECK:    test.with_optional_properties
// CHECK-SAME: anAttr = 0 simple = 1 nonTrivialStorage = "foo" hasDefault = some<0> nested = some<1>  longSyntax = some<"bar"> hasUnit maybeUnit = some<unit>
// GENERIC: "test.with_optional_properties"()
// GENERIC-SAME: <{anAttr = 0 : i32, hasDefault = [0], hasUnit, longSyntax = ["bar"], maybeUnit = [unit], nested = {{\[}}[1]], nonTrivialStorage = ["foo"], simple = [1]}> : () -> ()
test.with_optional_properties
  anAttr = 0
  simple = 1
  nonTrivialStorage = "foo"
  hasDefault = some<0>
  nested = some<1>
  longSyntax = some<"bar">
  hasUnit
  maybeUnit = some<unit>

// CHECK:    test.with_optional_properties
// CHECK-SAME: nested = some<none>
// GENERIC: "test.with_optional_properties"()
// GENERIC-SAME: nested = {{\[}}[]]
test.with_optional_properties nested = some<none>

// CHECK:    test.with_array_properties
// CHECK-SAME: ints = [1, 2] strings = ["a", "b"] nested = {{\[}}[1, 2], [3, 4]] opt = [-1, -2] explicitOptions = [none, 0] explicitUnits = [unit, unit_absent]
// GENERIC: "test.with_array_properties"()
test.with_array_properties ints = [1, 2] strings = ["a", "b"] nested = [[1, 2], [3, 4]] opt = [-1, -2] explicitOptions = [none, 0] explicitUnits = [unit, unit_absent] [] thats_has_default
