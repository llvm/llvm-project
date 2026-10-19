// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @st_async_global_sys
llvm.func @st_async_global_sys(%addr: !llvm.ptr<1>, %value: i32) {
  // CHECK: call void @llvm.nvvm.st.async.sys.i32(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, /* isMultimem= */ i1 false)
  nvvm.store.async.global %addr, %value scope = sys : !llvm.ptr<1>, i32
  llvm.return
}

// CHECK-LABEL: define void @st_async_global_gpu
llvm.func @st_async_global_gpu(%addr: !llvm.ptr<1>, %value: i32) {
  // CHECK: call void @llvm.nvvm.st.async.gpu.i32(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, /* isMultimem= */ i1 false)
  nvvm.store.async.global %addr, %value scope = gpu : !llvm.ptr<1>, i32
  llvm.return
}

// CHECK-LABEL: define void @st_async_global_multimem
llvm.func @st_async_global_multimem(%addr: !llvm.ptr<1>, %value: i32) {
  // CHECK: call void @llvm.nvvm.st.async.sys.i32(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, /* isMultimem= */ i1 true)
  nvvm.store.async.global %addr, %value scope = sys multimem = true : !llvm.ptr<1>, i32
  // CHECK: call void @llvm.nvvm.st.async.gpu.i32(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, /* isMultimem= */ i1 true)
  nvvm.store.async.global %addr, %value scope = gpu multimem = true : !llvm.ptr<1>, i32
  llvm.return
}

// CHECK-LABEL: define void @st_async_global_mmio
llvm.func @st_async_global_mmio(%addr: !llvm.ptr<1>, %value: i32) {
  // CHECK: call void @llvm.nvvm.st.async.mmio.sys.i32(ptr addrspace(1) %{{.*}}, i32 %{{.*}})
  nvvm.store.async.global %addr, %value scope = sys mmio = true : !llvm.ptr<1>, i32
  llvm.return
}

// CHECK-LABEL: define void @st_async_global_types
llvm.func @st_async_global_types(%addr: !llvm.ptr<1>, %v8: i8, %v16: i16, %v64: i64) {
  // CHECK: call void @llvm.nvvm.st.async.gpu.i8(ptr addrspace(1) %{{.*}}, i8 %{{.*}}, /* isMultimem= */ i1 false)
  nvvm.store.async.global %addr, %v8 scope = gpu : !llvm.ptr<1>, i8
  // CHECK: call void @llvm.nvvm.st.async.gpu.i16(ptr addrspace(1) %{{.*}}, i16 %{{.*}}, /* isMultimem= */ i1 false)
  nvvm.store.async.global %addr, %v16 scope = gpu : !llvm.ptr<1>, i16
  // CHECK: call void @llvm.nvvm.st.async.gpu.i64(ptr addrspace(1) %{{.*}}, i64 %{{.*}}, /* isMultimem= */ i1 false)
  nvvm.store.async.global %addr, %v64 scope = gpu : !llvm.ptr<1>, i64
  llvm.return
}
