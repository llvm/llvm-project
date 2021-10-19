// RUN: not mlir-opt %s -mlir-print-op-on-diagnostic 2>&1 | FileCheck %s

// This file tests the functionality of 'mlir-print-op-on-diagnostic'.

// CHECK: {{invalid to use 'test.invalid_attr'}}
// CHECK: {{see current operation: "builtin.module"()}}
module attributes {test.invalid_attr} {}
