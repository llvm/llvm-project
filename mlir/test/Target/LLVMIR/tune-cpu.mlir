// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @tune_cpu() #[[ATTRS:.*]] {
// CHECK: attributes #[[ATTRS]] = { "tune-cpu"="pentium4" }
llvm.func @tune_cpu() attributes {tune_cpu = "pentium4"} {
  llvm.return
}
