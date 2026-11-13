// RUN: mlir-opt %s --split-input-file --remove-dead-values="canonicalize=0" | FileCheck %s
// RUN: mlir-opt %s --split-input-file --remove-dead-values="canonicalize=0" --mlir-print-op-generic | FileCheck %s --check-prefix=GEN

// Remove all call arguments. Keep the operand bundle in its own segment.
// CHECK-LABEL: llvm.func internal @call_empty()
// CHECK-LABEL: llvm.func @caller(
// CHECK-SAME: %[[DEAD:[^:]+]]: i32, %[[TAG:[^:]+]]: i32
// CHECK: llvm.call @call_empty() ["tag"(%[[TAG]] : i32)] : () -> ()
// GEN: "llvm.call"
// GEN-SAME: callee = @call_empty
// GEN-SAME: op_bundle_sizes = array<i32: 1>
// GEN-SAME: operandSegmentSizes = array<i32: 0, 1>
llvm.func internal @call_empty(%dead: i32) attributes {sym_visibility = "private"} {
  llvm.return
}
llvm.func @caller(%dead: i32, %tag: i32) {
  llvm.call @call_empty(%dead) ["tag"(%tag : i32)] : (i32) -> ()
  llvm.return
}

// -----

// Remove disjoint arguments. Keep the middle argument and the bundle operand.
// CHECK-LABEL: llvm.func internal @call_partial(
// CHECK-SAME: %[[LIVE:[^:]+]]: i32)
// CHECK: llvm.call @use(%[[LIVE]]) : (i32) -> ()
// CHECK-LABEL: llvm.func @caller(
// CHECK-SAME: %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32, %[[C:[^:]+]]: i32, %[[TAG:[^:]+]]: i32
// CHECK: llvm.call @call_partial(%[[B]]) ["tag"(%[[TAG]] : i32)] : (i32) -> ()
// GEN: "llvm.call"{{.*}}callee = @call_partial
// GEN-SAME: op_bundle_sizes = array<i32: 1>
// GEN-SAME: operandSegmentSizes = array<i32: 1, 1>
llvm.func @use(i32)
llvm.func internal @call_partial(%a: i32, %b: i32, %c: i32) attributes {sym_visibility = "private"} {
  llvm.call @use(%b) : (i32) -> ()
  llvm.return
}
llvm.func @caller(%a: i32, %b: i32, %c: i32, %tag: i32) {
  llvm.call @call_partial(%a, %b, %c) ["tag"(%tag : i32)] : (i32, i32, i32) -> ()
  llvm.return
}

// -----

// Remove all invoke arguments when the other operand segments are empty.
// CHECK-LABEL: llvm.func internal @invoke_empty()
// CHECK-LABEL: llvm.func @caller(
// CHECK: llvm.invoke @invoke_empty() to ^{{.*}} unwind ^{{.*}} : () -> ()
// GEN: "llvm.invoke"
// GEN-SAME: callee = @invoke_empty
// GEN-SAME: operandSegmentSizes = array<i32: 0, 0, 0, 0>
llvm.func @__gxx_personality_v0(...) -> i32
llvm.func internal @invoke_empty(%dead: i32) attributes {sym_visibility = "private"} {
  llvm.return
}
llvm.func @caller(%dead: i32) attributes {personality = @__gxx_personality_v0} {
  llvm.invoke @invoke_empty(%dead) to ^normal unwind ^unwind : (i32) -> ()
^normal:
  llvm.return
^unwind:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
}

// -----

// Shrink the callee segment while keeping both successor segments and a bundle.
// CHECK-LABEL: llvm.func internal @invoke_partial(
// CHECK-SAME: %[[LIVE:[^:]+]]: i32)
// CHECK: llvm.call @use(%[[LIVE]]) : (i32) -> ()
// CHECK-LABEL: llvm.func @caller(
// CHECK-SAME: %[[DEAD:[^:]+]]: i32, %[[LIVE:[^:]+]]: i32, %[[NORMAL:[^:]+]]: i32, %[[UNWIND:[^:]+]]: i32, %[[TAG:[^:]+]]: i32
// CHECK: llvm.invoke @invoke_partial(%[[LIVE]]) to ^{{[^(]+}}(%[[NORMAL]] : i32) unwind ^{{[^(]+}}(%[[UNWIND]] : i32) ["tag"(%[[TAG]] : i32)] : (i32) -> ()
// GEN: "llvm.invoke"
// GEN-SAME: callee = @invoke_partial
// GEN-SAME: op_bundle_sizes = array<i32: 1>
// GEN-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1>
llvm.func @__gxx_personality_v0(...) -> i32
llvm.func @use(i32)
llvm.func internal @invoke_partial(%dead: i32, %live: i32) attributes {sym_visibility = "private"} {
  llvm.call @use(%live) : (i32) -> ()
  llvm.return
}
llvm.func @caller(%dead: i32, %live: i32, %normal: i32, %unwind: i32, %tag: i32) attributes {personality = @__gxx_personality_v0} {
  llvm.invoke @invoke_partial(%dead, %live) to ^normal(%normal : i32) unwind ^unwind(%unwind : i32) ["tag"(%tag : i32)] : (i32, i32) -> ()
^normal(%n: i32):
  llvm.call @use(%n) : (i32) -> ()
  llvm.return
^unwind(%u: i32):
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.call @use(%u) : (i32) -> ()
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
}
