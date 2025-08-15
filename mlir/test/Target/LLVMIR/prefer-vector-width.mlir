// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @prefer_vector_width() #[[ATTRS:.*]] {
// CHECK: attributes #[[ATTRS]] = { "prefer-vector-width"="128" }

llvm.func @prefer_vector_width() attributes {prefer_vector_width = "128"} {
  llvm.return
}
