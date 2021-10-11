// RUN: mlir-opt %s | mlir-opt -verify-diagnostics | FileCheck %s

//////////////
// Tests the types in the 'Test' dialect, not the ones in 'typedefs.mlir'

// CHECK: @simpleA(%arg0: !test.smpla)
func @simpleA(%A : !test.smpla) -> () {
  return
}

// CHECK: @compoundA(%arg0: !test.cmpnd_a<1, !test.smpla, [5, 6]>)
func @compoundA(%A : !test.cmpnd_a<1, !test.smpla, [5, 6]>)-> () {
  return
}

// CHECK: @testInt(%arg0: !test.int<signed, 8>, %arg1: !test.int<unsigned, 2>, %arg2: !test.int<none, 1>)
func @testInt(%A : !test.int<s, 8>, %B : !test.int<unsigned, 2>, %C : !test.int<n, 1>) {
  return
}

// CHECK: @structTest(%arg0: !test.struct<{field1,!test.smpla},{field2,!test.int<none, 3>}>)
func @structTest (%A : !test.struct< {field1, !test.smpla}, {field2, !test.int<none, 3>} > ) {
  return
}
