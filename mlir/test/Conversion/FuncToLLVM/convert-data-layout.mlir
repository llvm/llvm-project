// RUN: mlir-opt -convert-func-to-llvm %s | FileCheck %s
// RUN-32: mlir-opt -convert-func-to-llvm='data-layout=p:32:32:32' %s | FileCheck %s

// CHECK: module attributes {llvm.data_layout = ""}
// CHECK-32: module attributes {llvm.data_layout ="p:32:32:32"}
module {}
