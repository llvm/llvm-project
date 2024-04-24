// RUN: mlir-opt %s --verify-diagnostics
// This file is used as additional input.

// expected-error @below {{expected operation name in quotes}}
module {
