// # RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s
// # RUN: mlir-opt %s -mlir-print-op-generic -split-input-file  | mlir-opt -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// CHECK:   test.with_properties
// CHECK-SAME: a = 32, b = "foo", c = "bar", flag = true, array = [1, 2, 3, 4], array32 = [5, 6]{{$}}
// GENERIC:   "test.with_properties"()
// GENERIC-SAME: <{a = 32 : i64, array = array<i64: 1, 2, 3, 4>, array32 = array<i32: 5, 6>, b = "foo", c = "bar", flag = true}> : () -> ()
test.with_properties a = 32, b = "foo", c = "bar", flag = true, array = [1, 2, 3, 4], array32 = [5, 6]

// CHECK:   test.with_nice_properties
// CHECK-SAME:    "foo bar" is -3{{$}}
// GENERIC: "test.with_nice_properties"()
// GENERIC-SAME:  <{prop = {label = "foo bar", value = -3 : i32}}> : () -> ()
test.with_nice_properties "foo bar" is -3

// CHECK:   test.with_wrapped_properties
// CHECK-SAME:    <prop = "content for properties">{{$}}
// GENERIC: "test.with_wrapped_properties"()
// GENERIC-SAME:  <{prop = "content for properties"}> : () -> ()
test.with_wrapped_properties <{prop = "content for properties"}>

// CHECK: test.empty_properties
// GENERIC: "test.empty_properties"()
test.empty_properties

// An explicitly empty key-value list is also accepted.
// CHECK: test.empty_properties
// GENERIC: "test.empty_properties"()
test.empty_properties <>

// The key-value spelling uses the custom parsers and printers for both
// attributes and properties.
// CHECK: test.with_custom_prop_dict <prop = 2, attr = 1>
// GENERIC: "test.with_custom_prop_dict"()
// GENERIC-SAME: <{attr = 1 : i32, defaulted = 42 : i64, prop = 2 : i64, unit = false}>
test.with_custom_prop_dict <attr = 1, prop = 2>

// The generic DictionaryAttr spelling remains accepted for compatibility.
// CHECK: test.with_custom_prop_dict <prop = 4, attr = 3>
// GENERIC: "test.with_custom_prop_dict"()
// GENERIC-SAME: <{attr = 3 : i32, defaulted = 42 : i64, prop = 4 : i64, unit = false}>
test.with_custom_prop_dict <{attr = 3 : i32, prop = 4 : i64}>

// Entries are order-independent, and optional/default-valued entries use
// their custom parsers when present.
// CHECK: test.with_custom_prop_dict <prop = 6, defaulted = 43, attr = 5, optional = "set">
// GENERIC: "test.with_custom_prop_dict"()
// GENERIC-SAME: <{attr = 5 : i32, defaulted = 43 : i64, optional = "set", prop = 6 : i64, unit = false}>
test.with_custom_prop_dict <optional = "set", defaulted = 43, prop = 6, attr = 5>

// A field name that is also the start of an attribute must not be consumed by
// the legacy DictionaryAttr compatibility probe.
// CHECK: test.with_custom_prop_dict <prop = 8, unit = unit, attr = 7>
// GENERIC: "test.with_custom_prop_dict"()
// GENERIC-SAME: <{attr = 7 : i32, defaulted = 42 : i64, prop = 8 : i64, unit}>
test.with_custom_prop_dict <unit = unit, attr = 7, prop = 8>

// Inherent attributes use their custom assembly printer in the key-value
// spelling. Optional enum attributes compile and are omitted when absent.
// CHECK: test.with_custom_attr_prop_dict <prop = 9, attr = first>
test.with_custom_attr_prop_dict <attr = first, prop = 9>
// CHECK: test.with_custom_attr_prop_dict <prop = 10, attr = first, optionalAttr = second>
test.with_custom_attr_prop_dict <optionalAttr = second, prop = 10, attr = first>

// Properties bound elsewhere in the assembly format are excluded from the
// key-value list.
// CHECK: test.with_properties_and_attr 7 <rhs = 8>
// GENERIC: "test.with_properties_and_attr"()
// GENERIC-SAME: <{lhs = 7 : i32, rhs = 8 : i64}>
test.with_properties_and_attr 7 <rhs = 8>

// A property without a usable custom parser falls back to its attribute
// conversion for this compatibility spelling.
// CHECK: test.with_wrapped_properties <prop = "custom spelling">
// GENERIC: "test.with_wrapped_properties"()
// GENERIC-SAME: <{prop = "custom spelling"}>
test.with_wrapped_properties <prop = "custom spelling">

// Forwarding property wrappers preserve whether their base uses the default
// FieldParser, so a wrapped custom storage type still uses attribute fallback.
// CHECK: test.with_default_wrapped_properties
// GENERIC: "test.with_default_wrapped_properties"()
// GENERIC-SAME: <{prop = "wrapped default spelling"}>
test.with_default_wrapped_properties <prop = "wrapped default spelling">

// A container FieldParser is unavailable when its element parser is
// unavailable, so the complete property also falls back to conversion.
// CHECK: test.with_wrapped_array_properties
// GENERIC: "test.with_wrapped_array_properties"()
// GENERIC-SAME: <{prop = ["first", "second"]}>
test.with_wrapped_array_properties <prop = ["first", "second"]>

// Default optional and container FieldParsers do not delimit exactly one
// property value, so they use attribute conversion in a key-value list. The
// following scalar key also checks that the container does not consume the
// outer comma.
// CHECK: test.with_key_value_parser_boundaries
// CHECK-SAME: <values = array<i64: 1, 2>, maybe = [], maybeEnum = [],
// CHECK-SAME: specializedValues = [3, 4], specializedMaybe = some<7>, next = 9>
// GENERIC: "test.with_key_value_parser_boundaries"()
// GENERIC-SAME: <{maybe = [], maybeEnum = [], next = 9 : i64, specializedMaybe = [7 : i16], specializedValues = array<i32: 3, 4>, values = array<i64: 1, 2>}>
test.with_key_value_parser_boundaries <specializedValues = [3, 4], specializedMaybe = some<7>, values = array<i64: 1, 2>, maybe = [], maybeEnum = [], next = 9>

// A comma-separated bit-enum FieldParser is not compositional with the outer
// list, so prop-dict uses its attribute conversion before parsing another key.
// CHECK: test.op_with_bit_enum_prop_dict
// CHECK-SAME: <flags = 3 : i32, next = 9>
// GENERIC: "test.op_with_bit_enum_prop_dict"()
// GENERIC-SAME: <{flags = 3 : i32, next = 9 : i64}>
test.op_with_bit_enum_prop_dict <flags = 3 : i32, next = 9>

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

// Tests that the variadic segment size properties survive a round-trip
// through the *custom* (non-generic) parser/printer when the assembly format
// uses a bulk `functional-type(operands, results)` directive, which prevents
// the printer from eliding `operandSegmentSizes` / `resultSegmentSizes` from
// `<{...}>`. Without the parser-side fix, re-parsing the CHECK line below
// (which is exactly what the printer emits) fails with "duplicate or unknown
// key 'operandSegmentSizes' in dictionary attribute".
// CHECK: test.variadic_segment_prop_bulk_type(%[[CI64]], %[[CI64]], %[[CI64]]) : (i64, i64, i64) -> (i64, i64, i64) <operandSegmentSizes = [2, 1], resultSegmentSizes = [2, 1]>
// GENERIC: "test.variadic_segment_prop_bulk_type"(%[[CI64]], %[[CI64]], %[[CI64]]) <{operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 2, 1>}> : (i64, i64, i64) -> (i64, i64, i64)
test.variadic_segment_prop_bulk_type(%ci64, %ci64, %ci64) : (i64, i64, i64) -> (i64, i64, i64) <operandSegmentSizes = [2, 1], resultSegmentSizes = [2, 1]>

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
// GENERIC-SAME:  <{hasDefault = [], hasUnit = false, longSyntax = [], maybeUnit = [], nested = [], nonTrivialStorage = [], simple = [0], simplei8 = [], simpleui8 = []}> : () -> ()
test.with_optional_properties simple = 0

// CHECK:   test.with_optional_properties
// CHECK-SAME: simple = 1 simplei8 = -1 simpleui8 = 255
// GENERIC: "test.with_optional_properties"()
// GENERIC-SAME:  <{hasDefault = [], hasUnit = false, longSyntax = [], maybeUnit = [], nested = [], nonTrivialStorage = [], simple = [1], simplei8 = [-1 : i8], simpleui8 = [-1 : i8]}> : () -> ()
test.with_optional_properties simple = 1 simplei8 = -1 simpleui8 = 255

// CHECK:   test.with_optional_properties{{$}}
// GENERIC: "test.with_optional_properties"()
// GENERIC-SAME: simple = []
test.with_optional_properties

// CHECK:    test.with_optional_properties
// CHECK-SAME: anAttr = 0 simple = 1 nonTrivialStorage = "foo" hasDefault = some<0> nested = some<1>  longSyntax = some<"bar"> hasUnit maybeUnit = some<unit>
// GENERIC: "test.with_optional_properties"()
// GENERIC-SAME: <{anAttr = 0 : i32, hasDefault = [0], hasUnit, longSyntax = ["bar"], maybeUnit = [unit], nested = {{\[}}[1]], nonTrivialStorage = ["foo"], simple = [1], simplei8 = [], simpleui8 = []}> : () -> ()
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

// Tests that DefaultValuedProp is elided from prop-dict when value equals default.
// CHECK: test.op_with_property_predicates
// CHECK-SAME: <scalar = 1, more_constrained = 1, array = [],
// CHECK-SAME: non_empty_unconstrained = [1], non_empty_constrained = [1], unconstrained = 0>
// CHECK-NOT: defaulted
test.op_with_property_predicates <{
  scalar = 1 : i64,
  more_constrained = 1 : i64,
  array = [],
  non_empty_unconstrained = [1],
  non_empty_constrained = [1],
  unconstrained = 0 : i64}>

// Keyed parsing composes optional and aggregate property parsers with a
// following outer dictionary entry.
// CHECK: test.op_with_property_predicates
// CHECK-SAME: optional = 2
// CHECK-SAME: array = [3, 4]
test.op_with_property_predicates <
  scalar = 1,
  optional = 2,
  more_constrained = 1,
  array = [3, 4],
  non_empty_unconstrained = [1],
  non_empty_constrained = [1],
  unconstrained = 0>

// Tests that DefaultValuedProp is printed when value differs from default.
// CHECK: test.op_with_property_predicates
// CHECK-SAME: defaulted = 3
test.op_with_property_predicates <{
  scalar = 1 : i64,
  defaulted = 3 : i64,
  more_constrained = 1 : i64,
  array = [],
  non_empty_unconstrained = [1],
  non_empty_constrained = [1],
  unconstrained = 0 : i64}>
