// RUN: mlir-opt %s --emit-bytecode > %T/versioning-fallback.mlirbc
"test.versionedD"() <{
  attribute = #test.compound_attr_no_reading<
    noReadingNested = #test.compound_attr_no_reading_nested<
      value = "foo", 
      payload = [24, "bar"]
    >, 
    supportsReading = #test.attr_params<42, 24>
  >
}> : () -> ()

// COM: check that versionedD was parsed as a fallback op.
// RUN: mlir-opt %T/versioning-fallback.mlirbc | FileCheck %s --check-prefix=CHECK-PARSE
// CHECK-PARSE: test.bytecode.fallback 
// CHECK-PARSE-SAME: encodedReqdAttributes = [#test.bytecode_fallback<attrIndex = 100, 
// CHECK-PARSE-SAME:  encodedReqdAttributes = [#test.bytecode_fallback<attrIndex = 101, 
// CHECK-PARSE-SAME:    encodedReqdAttributes = ["foo", [24, "bar"]], 
// CHECK-PARSE-SAME:  #test.attr_params<42, 24>
// CHECK-PARSE-SAME: opname = "test.versionedD", 
// CHECK-PARSE-SAME: opversion = 1

// COM: check that the bytecode roundtrip was successful
// RUN: mlir-opt %T/versioning-fallback.mlirbc --verify-roundtrip

// COM: check that the bytecode roundtrip is bitwise exact
// RUN: mlir-opt %T/versioning-fallback.mlirbc --emit-bytecode | diff %T/versioning-fallback.mlirbc -
