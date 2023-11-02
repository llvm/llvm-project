// RUN: mlir-opt -set-llvm-module-datalayout -convert-func-to-llvm %s | FileCheck %s

// RUN-32: mlir-opt -set-llvm-module-datalayout='data-layout=p:32:32:32' -convert-func-to-llvm %s \
// RUN-32: | FileCheck %s

// CHECK: module attributes {llvm.data_layout = ""}
// CHECK-32: module attributes {llvm.data_layout ="p:32:32:32"}
module {}
