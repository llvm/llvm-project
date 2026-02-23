// RUN: mlir-translate --split-input-file -mlir-to-llvmir %s | FileCheck %s

module {
  llvm.func spir_funccc @_Z8prefetchPU3AS1Kcm(!llvm.ptr<1>, i64)

  llvm.func @prefetch(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    // CHECK: @[[CACHECONTROL:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\220,0\22}\00", section "llvm.metadata"
    // CHECK: @[[CACHECONTROL1:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\221,0\22}\00", section "llvm.metadata"
    // CHECK-LABEL: define void @prefetch
    // CHECK-SAME: ptr addrspace(1) %[[ARG0:.*]])
    // CHECK: %[[ANNOTATED:.*]] = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %[[ARG0]], ptr addrspace(1) @[[CACHECONTROL]],
    // CHECK: %[[ANNOTATED1:.*]] = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %[[ANNOTATED]], ptr addrspace(1) @[[CACHECONTROL1]],
    // CHECK: call spir_func void @_Z8prefetchPU3AS1Kcm(ptr addrspace(1) %[[ANNOTATED1]], i64 1)
    llvm.call spir_funccc @_Z8prefetchPU3AS1Kcm(%arg0, %0) {function_type = !llvm.func<void (ptr<1>, i64)>, linkage = #llvm.linkage<external>, no_unwind, sym_name = "_Z8prefetchPU3AS1Kcm", visibility_ = 0 : i64, xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32, 0 : i32], [6442 : i32, 1 : i32, 1 : i32, 0 : i32]]} : (!llvm.ptr<1>, i64) -> ()
    llvm.return
  }
}

// -----

module {
  // CHECK: @[[CACHECONTROL:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\220,0\22}\00", section "llvm.metadata"
  // CHECK: @[[CACHECONTROL1:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\221,0\22}\00", section "llvm.metadata"
  // CHECK-LABEL: define i32 @load
  // CHECK-SAME: ptr addrspace(1) %[[ARG0:.*]])
  llvm.func @load(%arg0: !llvm.ptr<1>) -> i32 {
    // CHECK: %[[ANNOTATED:.*]] = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %[[ARG0]], ptr addrspace(1) @[[CACHECONTROL]],
    // CHECK: %[[ANNOTATED1:.*]] = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %[[ANNOTATED]], ptr addrspace(1) @[[CACHECONTROL1]],
    // CHECK: load i32, ptr addrspace(1) %[[ANNOTATED1]], align 4
    %0 = llvm.load %arg0 {xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32, 0 : i32], [6442 : i32, 1 : i32, 1 : i32, 0 : i32]]} : !llvm.ptr<1> -> i32
    llvm.return %0 : i32
  }
}

// -----

module {
  // CHECK: @[[CACHECONTROL:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6443:\220,0\22}\00", section "llvm.metadata"
  // CHECK: @[[CACHECONTROL1:.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{6443:\221,0\22}\00", section "llvm.metadata"
  // CHECK-LABEL: define void @store
  // CHECK-SAME: ptr addrspace(1) %[[ARG0:.*]], i32 %[[ARG1:.*]])
  llvm.func @store(%arg0: !llvm.ptr<1>, %arg1: i32) {
    // CHECK: %[[ANNOTATED:.*]] = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %[[ARG0]], ptr addrspace(1) @[[CACHECONTROL]],
    // CHECK: %[[ANNOTATED1:.*]] = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %[[ANNOTATED]], ptr addrspace(1) @[[CACHECONTROL1]],
    // CHECK: store i32 %[[ARG1]], ptr addrspace(1) %[[ANNOTATED1]], align 4
    llvm.store %arg1, %arg0 {xevm.DecorationCacheControl = [[6443 : i32, 0 : i32, 2 : i32, 0 : i32], [6443 : i32, 1 : i32, 2 : i32, 0 : i32]]} : i32, !llvm.ptr<1>
    llvm.return
  }
}
