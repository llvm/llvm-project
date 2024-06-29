// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @target_cpu() #[[ATTRS:.*]] {
// CHECK: attributes #[[ATTRS]] = { "target-cpu"="gfx90a" }
llvm.func @target_cpu() attributes {target_cpu = "gfx90a"} {
  llvm.return
}
