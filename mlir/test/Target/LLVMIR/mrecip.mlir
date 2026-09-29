// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @reciprocal_estimates() #[[ATTRS:.*]] {
// CHECK: attributes #[[ATTRS]] = { "reciprocal-estimates"="all" }

llvm.func @reciprocal_estimates() attributes {reciprocal_estimates = "all"} {
  llvm.return
}
