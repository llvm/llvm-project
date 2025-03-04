// RUN: mlir-opt %s --emit-bytecode > %T/versioning-fallback.mlirbc
"test.versionedD"() <{attribute = #test.attr_params<42, 24>}> : () -> ()

// COM: check that versionedD was parsed as a fallback op.
// RUN: mlir-opt %T/versioning-fallback.mlirbc | FileCheck %s --check-prefix=CHECK-PARSE
// CHECK-PARSE: test.bytecode.fallback 
// CHECK-PARSE-SAME: opname = "test.versionedD"

// COM: check that the bytecode roundtrip was successful
// RUN: mlir-opt %T/versioning-fallback.mlirbc --verify-roundtrip

// COM: check that the bytecode roundtrip is bitwise exact
// RUN: mlir-opt %T/versioning-fallback.mlirbc --emit-bytecode | diff %T/versioning-fallback.mlirbc -
