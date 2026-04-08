// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @use_sample_profile() #[[ATTRS:.*]] {
// CHECK: attributes #[[ATTRS]] = { "use-sample-profile" }
llvm.func @use_sample_profile() attributes {use_sample_profile = true} {
  llvm.return
}
