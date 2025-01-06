// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: !llvm.ident = !{![[ID:[0-9]+]]}
// CHECK: ![[ID]] = !{!"flang version 61.7.4"}
module attributes {llvm.ident = "flang version 61.7.4"} {
}
