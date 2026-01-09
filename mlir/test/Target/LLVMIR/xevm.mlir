// RUN: mlir-translate --split-input-file -mlir-to-llvmir %s | FileCheck %s

module {
  // CHECK-LABEL: define i32 @load(ptr addrspace(1)
  // CHECK-SAME: %[[ARG0:.*]]) {
  llvm.func @load(%arg0: !llvm.ptr<1>) -> i32 {
    // CHECK: getelementptr i32
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    %0 = llvm.getelementptr %arg0[0] {xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32], [6442 : i32, 1 : i32, 1 : i32]]} : (!llvm.ptr<1>) -> !llvm.ptr<1>, i32
    %1 = llvm.load %0 : !llvm.ptr<1> -> i32
    llvm.return %1 : i32
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6442, i32 0, i32 1}
// CHECK: ![[DECO3]] = !{i32 6442, i32 1, i32 1}

// -----
module {
  // CHECK-LABEL: define void @store(ptr addrspace(1)
  // CHECK-SAME: %[[ARG0:.*]], i32 %[[ARG1:.*]]) {
  llvm.func @store(%arg0: !llvm.ptr<1>, %arg1: i32) {
    // CHECK: getelementptr i32
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    %0 = llvm.getelementptr %arg0[0] {xevm.DecorationCacheControl = [[6443 : i32, 0 : i32, 1 : i32], [6443 : i32, 1 : i32, 2 : i32]]} : (!llvm.ptr<1>) -> !llvm.ptr<1>, i32
    llvm.store %arg1, %0 : i32, !llvm.ptr<1>
    llvm.return
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6443, i32 0, i32 1}
// CHECK: ![[DECO3]] = !{i32 6443, i32 1, i32 2}

