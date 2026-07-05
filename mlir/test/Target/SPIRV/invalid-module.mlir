// RUN: mlir-translate %s -serialize-spirv -no-implicit-module -verify-diagnostics

// expected-error@below {{expected a 'spirv.module' op, got 'builtin.module'}}
module {}
