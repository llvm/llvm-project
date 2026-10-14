// RUN: mlir-opt %s -inline -symbol-dce -canonicalize='region-simplify=aggressive' -cse -sccp | FileCheck %s
// RUN: mlir-opt %s -inline -symbol-dce -canonicalize='region-simplify=aggressive' -cse -sccp | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// Inlining must preserve immediate attributes when cloning the callee and
// remapping its SSA arguments, including operand bundle inputs.
llvm.func internal @callee(%x: i32, %extra: i64) -> i32 attributes {sym_visibility = "private"} {
  %r = llvm.call_intrinsic "llvm.ctlz"(%x) ["tag"(%extra : i64)] {
    immarg_positions = array<i32: 1>, immarg_values = [false]
  } : (i32) -> i32
  llvm.return %r : i32
}

// CHECK-NOT: llvm.func internal @callee
// CHECK-LABEL: llvm.func @inline_immediates(
// CHECK-SAME: %[[X:.*]]: i32, %[[EXTRA:.*]]: i64)
// CHECK-NOT: llvm.call @callee
// CHECK: %[[FIRST:.*]] = llvm.call_intrinsic "llvm.ctlz"(%[[X]]) ["tag"(%[[EXTRA]] : i64)] {immarg_positions = array<i32: 1>, immarg_values = [false]}
// CHECK: %[[SECOND:.*]] = llvm.call_intrinsic "llvm.ctlz"(%[[FIRST]]) ["tag"(%[[EXTRA]] : i64)] {immarg_positions = array<i32: 1>, immarg_values = [false]}
// CHECK: llvm.return %[[SECOND]] : i32
// LLVM-LABEL: define i32 @inline_immediates(
// LLVM-SAME: i32 %[[X:.*]], i64 %[[EXTRA:.*]])
// LLVM: %[[FIRST:.*]] = call i32 @llvm.ctlz.i32(i32 %[[X]], i1 false) [ "tag"(i64 %[[EXTRA]]) ]
// LLVM: %[[SECOND:.*]] = call i32 @llvm.ctlz.i32(i32 %[[FIRST]], i1 false) [ "tag"(i64 %[[EXTRA]]) ]
// LLVM: ret i32 %[[SECOND]]
llvm.func @inline_immediates(%x: i32, %extra: i64) -> i32 {
  %a = llvm.call @callee(%x, %extra) : (i32, i64) -> i32
  %b = llvm.call @callee(%a, %extra) : (i32, i64) -> i32
  llvm.return %b : i32
}
