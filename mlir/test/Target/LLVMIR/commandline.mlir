// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: !llvm.commandline = !{![[S:[0-9]+]]}
// CHECK: ![[S]] = !{!"exec -o infile"}
module attributes {llvm.commandline = "exec -o infile"} {
}
