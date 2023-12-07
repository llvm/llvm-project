// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// CHECK: module
// CHECK: foo.asdf = 0 : i0
module attributes { foo.asdf = 0 : i0 } { }
