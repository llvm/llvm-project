// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {llvm.module_asm = ["foo", "bar"]} {}

// CHECK: module asm "foo"
// CHECK: module asm "bar"
