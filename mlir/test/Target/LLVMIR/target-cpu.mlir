// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @target_cpu
// CHECK: attributes #{{.*}} = { "target-cpu"="gfx90a" }
llvm.func @target_cpu() attributes {target_cpu = "gfx90a"} {
  llvm.return
}
