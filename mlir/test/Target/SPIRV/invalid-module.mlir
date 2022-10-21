// RUN: mlir-translate %s -serialize-spirv -no-implicit-module -verify-diagnostics

// expected-error@below {{expected a 'builtin.module' op, got 'spirv.module'}}
spirv.module Logical Simple {}
