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

