// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @nvvm_wmma_load_tf32
func.func @nvvm_wmma_load_tf32(%arg0: !llvm.ptr<i32>, %arg1 : i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: nvvm.wmma.load {{.*}} {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<i32>) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %0 : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @cp_async
llvm.func @cp_async(%arg0: !llvm.ptr<i8, 3>, %arg1: !llvm.ptr<i8, 1>) {
// CHECK:  nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, cache =  ca
  nvvm.cp.async.shared.global %arg0, %arg1, 16, cache=ca : !llvm.ptr<i8, 3>, !llvm.ptr<i8, 1>
// CHECK:  nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, cache =  cg 
  nvvm.cp.async.shared.global %arg0, %arg1, 16, cache=cg : !llvm.ptr<i8, 3>, !llvm.ptr<i8, 1>
// CHECK: nvvm.cp.async.commit.group
  nvvm.cp.async.commit.group
// CHECK: nvvm.cp.async.wait.group 0
  nvvm.cp.async.wait.group 0
  llvm.return
}

// CHECK-LABEL: llvm.func @ld_matrix
llvm.func @ld_matrix(%arg0: !llvm.ptr<i32, 3>) {
  // CHECK: nvvm.ldmatrix %{{.*}} {layout = #nvvm.mma_layout<row>, num = 1 : i32} : (!llvm.ptr<i32, 3>) -> i32
  %l1 = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> i32
  // CHECK: nvvm.ldmatrix %{{.*}} {layout = #nvvm.mma_layout<row>, num = 2 : i32} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32)>
  %l2 = nvvm.ldmatrix %arg0 {num = 2 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32)>
  // CHECK: nvvm.ldmatrix %{{.*}} {layout = #nvvm.mma_layout<row>, num = 4 : i32} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
  %l4 = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return
}

// CHECK-LABEL: llvm.func @redux_sync
llvm.func @redux_sync(%value : i32, %offset : i32) -> i32 {
  // CHECK: nvvm.redux.sync  add %{{.*}}
  %r1 = nvvm.redux.sync add %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  max %{{.*}}
  %r2 = nvvm.redux.sync max %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  min %{{.*}}
  %r3 = nvvm.redux.sync min %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  umax %{{.*}}
  %r5 = nvvm.redux.sync umax %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  umin %{{.*}}
  %r6 = nvvm.redux.sync umin %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  and %{{.*}}
  %r7 = nvvm.redux.sync and %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  or %{{.*}}
  %r8 = nvvm.redux.sync or %value, %offset : i32 -> i32
  // CHECK: nvvm.redux.sync  xor %{{.*}}
  %r9 = nvvm.redux.sync xor %value, %offset : i32 -> i32
  llvm.return %r1 : i32
}
