// RUN: mlir-opt -test-type-interfaces -allow-unregistered-dialect -verify-diagnostics %s

// expected-remark@below {{'!test.test_type' - TestA}}
// expected-remark@below {{'!test.test_type' - TestB}}
// expected-remark@below {{'!test.test_type' - TestC}}
// expected-remark@below {{'!test.test_type' - 42 - Int TestC}}
// expected-remark@below {{'!test.test_type' - 3.140000e+00 - Float TestC}}
// expected-remark@below {{'!test.test_type' - TestD}}
// expected-remark@below {{'!test.test_type' - TestRet}}
// expected-remark@below {{'!test.test_type' - TestE}}
%foo0 = "foo.test"() : () -> (!test.test_type)

// Type without the test interface.
%foo1 = "foo.test"() : () -> (i32)
