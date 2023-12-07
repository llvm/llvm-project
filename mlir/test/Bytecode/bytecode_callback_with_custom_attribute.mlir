// RUN: mlir-opt %s -split-input-file --test-bytecode-callback="callback-test=3" | FileCheck %s --check-prefix=TEST_3
// RUN: mlir-opt %s -split-input-file --test-bytecode-callback="callback-test=4" | FileCheck %s --check-prefix=TEST_4

"test.versionedC"() <{attribute = #test.attr_params<42, 24>}> : () -> ()

// TEST_3: Overriding TestAttrParamsAttr encoding...
// TEST_3: "test.versionedC"() <{attribute = dense<[42, 24]> : tensor<2xi32>}> : () -> ()

// -----

"test.versionedC"() <{attribute = dense<[42, 24]> : tensor<2xi32>}> : () -> ()

// TEST_4: Overriding parsing of TestAttrParamsAttr encoding...
// TEST_4: "test.versionedC"() <{attribute = #test.attr_params<42, 24>}> : () -> ()
