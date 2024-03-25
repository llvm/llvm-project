// RUN: mlir-opt %s --test-transform-dialect-interpreter=test-module-generation=1 --verify-diagnostics

// expected-remark @below {{remark from generated}}
module {}
