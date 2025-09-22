// RUN: mlir-translate --split-input-file -mlir-to-llvmir %s | FileCheck %s

module {
  llvm.func spir_funccc @_Z8prefetchPU3AS1Kcm(!llvm.ptr<1>, i64)
  llvm.func @prefetch(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    // CHECK-LABEL: call spir_func void @_Z8prefetchPU3AS1Kcm
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    llvm.call spir_funccc @_Z8prefetchPU3AS1Kcm(%arg0, %0)
      {function_type = !llvm.func<void (ptr<1>, i64)>, linkage = #llvm.linkage<external>,
       no_unwind, sym_name = "_Z8prefetchPU3AS1Kcm", visibility_ = 0 : i64,
       xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32, 0 : i32], [6442 : i32, 1 : i32, 1 : i32, 0 : i32]]}
      : (!llvm.ptr<1>, i64) -> ()
    llvm.return
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6442, i32 0, i32 1, i32 0}
// CHECK: ![[DECO3]] = !{i32 6442, i32 1, i32 1, i32 0}

// -----
module {
  // CHECK-LABEL: define i32 @load(ptr addrspace(1)
  // CHECK-SAME: %[[ARG0:.*]]) {
  llvm.func @load(%arg0: !llvm.ptr<1>) -> i32 {
    // CHECK: load i32, ptr addrspace(1) %[[ARG0]], align 4,
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    %0 = llvm.load %arg0 {xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32, 0 : i32], [6442 : i32, 1 : i32, 1 : i32, 0 : i32]]} : !llvm.ptr<1> -> i32
    llvm.return %0 : i32
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6442, i32 0, i32 1, i32 0}
// CHECK: ![[DECO3]] = !{i32 6442, i32 1, i32 1, i32 0}

// -----
module {
  // CHECK-LABEL: define void @store(ptr addrspace(1)
  // CHECK-SAME: %[[ARG0:.*]], i32 %[[ARG1:.*]]) {
  llvm.func @store(%arg0: !llvm.ptr<1>, %arg1: i32) {
    // CHECK: store i32 %[[ARG1]], ptr addrspace(1) %[[ARG0]], align 4,
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    llvm.store %arg1, %arg0 {xevm.DecorationCacheControl = [[6443 : i32, 0 : i32, 2 : i32, 0 : i32], [6443 : i32, 1 : i32, 2 : i32, 0 : i32]]} : i32, !llvm.ptr<1>
    llvm.return
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6443, i32 0, i32 2, i32 0}
// CHECK: ![[DECO3]] = !{i32 6443, i32 1, i32 2, i32 0}

