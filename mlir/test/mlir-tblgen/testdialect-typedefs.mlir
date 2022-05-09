// RUN: mlir-opt %s | mlir-opt -verify-diagnostics | FileCheck %s

//////////////
// Tests the types in the 'Test' dialect, not the ones in 'typedefs.mlir'

// CHECK: @simpleA(%arg0: !test.smpla)
func.func @simpleA(%A : !test.smpla) -> () {
  return
}

// CHECK: @compoundA(%arg0: !test.cmpnd_a<1, !test.smpla, [5, 6]>)
func.func @compoundA(%A : !test.cmpnd_a<1, !test.smpla, [5, 6]>)-> () {
  return
}

// CHECK: @compoundNested(%arg0: !test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>)
func.func @compoundNested(%arg0: !test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>) -> () {
  return
}

// Same as above, but we're parsing the complete spec for the inner type
// CHECK: @compoundNestedExplicit(%arg0: !test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>)
func.func @compoundNestedExplicit(%arg0: !test.cmpnd_nested_outer<i !test.cmpnd_inner<42 <1, !test.smpla, [5, 6]>>>) -> () {
// Verify that the type prefix is elided and optional
// CHECK: format_cpmd_nested_type %arg0 nested <i <42 <1, !test.smpla, [5, 6]>>>
// CHECK: format_cpmd_nested_type %arg0 nested <i <42 <1, !test.smpla, [5, 6]>>>
  test.format_cpmd_nested_type %arg0 nested !test.cmpnd_nested_outer<i !test.cmpnd_inner<42 <1, !test.smpla, [5, 6]>>>
  test.format_cpmd_nested_type %arg0 nested <i <42 <1, !test.smpla, [5, 6]>>>
  return
}

// CHECK-LABEL: @compoundNestedQual
// CHECK-SAME: !test.cmpnd_nested_outer_qual<i !test.cmpnd_inner<42 <1, !test.smpla, [5, 6]>>>
func.func private @compoundNestedQual(%arg0: !test.cmpnd_nested_outer_qual<i !test.cmpnd_inner<42 <1, !test.smpla, [5, 6]>>>) -> ()

// CHECK: @testInt(%arg0: !test.int<signed, 8>, %arg1: !test.int<unsigned, 2>, %arg2: !test.int<none, 1>)
func.func @testInt(%A : !test.int<s, 8>, %B : !test.int<unsigned, 2>, %C : !test.int<n, 1>) {
  return
}

// CHECK: @structTest(%arg0: !test.struct<{field1,!test.smpla}, {field2,!test.int<none, 3>}>)
func.func @structTest (%A : !test.struct< {field1, !test.smpla}, {field2, !test.int<none, 3>} > ) {
  return
}
