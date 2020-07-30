// RUN: mlir-opt -test-type-interfaces -allow-unregistered-dialect -verify-diagnostics %s

// expected-remark@below {{'!test.test_type' - TestA}}
// expected-remark@below {{'!test.test_type' - TestB}}
// expected-remark@below {{'!test.test_type' - TestC}}
// expected-remark@below {{'!test.test_type' - TestD}}
// expected-remark@below {{'!test.test_type' - TestE}}
%foo0 = "foo.test"() : () -> (!test.test_type)

// Type without the test interface.
%foo1 = "foo.test"() : () -> (i32)
