// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @func_with_empty_named_info()
// Check that translation doens't crash in the presence of an inlineble call
// with a named loc that has no backing source info.
llvm.func @callee() {
  llvm.return
} loc("calleesource.cc":1:1)
llvm.func @func_with_empty_named_info() {
  llvm.call @callee() : () -> () loc("named with no line info")
  llvm.return
}

// CHECK-LABEL: define void @func_no_debug()
// CHECK-NOT: !dbg
llvm.func @func_no_debug() {
  llvm.return loc(unknown)
} loc(unknown)

// CHECK-LABEL: define void @func_with_debug()
// CHECK-SAME: !dbg ![[FUNC_LOC:[0-9]+]]
llvm.func @func_with_debug() {
  // CHECK: call void @func_no_debug(), !dbg ![[CALLSITE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(callsite("mysource.cc":3:4 at "mysource.cc":5:6))

  // CHECK: call void @func_no_debug(), !dbg ![[FILE_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("foo.mlir":1:2)

  // CHECK: call void @func_no_debug(), !dbg ![[NAMED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc("named"("foo.mlir":10:10))

  // CHECK: call void @func_no_debug(), !dbg ![[FUSED_LOC:[0-9]+]]
  llvm.call @func_no_debug() : () -> () loc(fused[callsite("mysource.cc":1:1 at "mysource.cc":5:6), "mysource.cc":1:1])

  llvm.return
} loc("foo.mlir":1:1)

// CHECK-DAG: ![[FUNC_LOC]] = distinct !DISubprogram{{.*}}, line: 1
// CHECK-DAG: ![[CALLSITE_LOC]] = !DILocation(line: 3, column: 4,
// CHECK-DAG: ![[FILE_LOC]] = !DILocation(line: 1, column: 2,
// CHECK-DAG: ![[NAMED_LOC]] = !DILocation(line: 10, column: 10
// CHECK-DAG: ![[FUSED_LOC]] = !DILocation(line: 1, column: 1
