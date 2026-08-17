// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @disable_tail_calls() #[[ATTRS:.*]] {
// CHECK: attributes #[[ATTRS]] = { "disable-tail-calls"="true" }
llvm.func @disable_tail_calls() attributes {disable_tail_calls = true} {
  llvm.return
}
