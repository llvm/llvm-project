// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @disable_tail_calls() #[[ATTRS_TRUE:.*]] {
llvm.func @disable_tail_calls() attributes {disable_tail_calls = true} {
  llvm.return
}

// CHECK: define void @disable_tail_calls_false() #[[ATTRS_FALSE:.*]] {
llvm.func @disable_tail_calls_false() attributes {disable_tail_calls = false} {
  llvm.return
}

// CHECK: attributes #[[ATTRS_TRUE]] = { "disable-tail-calls"="true" }
// CHECK: attributes #[[ATTRS_FALSE]] = { "disable-tail-calls"="false" }
