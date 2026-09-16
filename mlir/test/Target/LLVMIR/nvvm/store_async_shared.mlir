// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @st_async_shared_cluster
llvm.func @st_async_shared_cluster(%addr: !llvm.ptr<7>, %value: i32, %mbar: !llvm.ptr<7>) {
  // CHECK: call void @llvm.nvvm.st.async.i32(ptr addrspace(7) %{{.*}}, i32 %{{.*}}, ptr addrspace(7) %{{.*}})
  nvvm.store.async.shared %addr, %value, mbarrier = %mbar : !llvm.ptr<7>, i32, !llvm.ptr<7>
  llvm.return
}

// CHECK-LABEL: define void @st_async_shared_cluster_types
llvm.func @st_async_shared_cluster_types(%addr: !llvm.ptr<7>, %v64: i64, %v128: i128, %mbar: !llvm.ptr<7>) {
  // CHECK: call void @llvm.nvvm.st.async.i64(ptr addrspace(7) %{{.*}}, i64 %{{.*}}, ptr addrspace(7) %{{.*}})
  nvvm.store.async.shared %addr, %v64, mbarrier = %mbar : !llvm.ptr<7>, i64, !llvm.ptr<7>
  // CHECK: call void @llvm.nvvm.st.async.i128(ptr addrspace(7) %{{.*}}, i128 %{{.*}}, ptr addrspace(7) %{{.*}})
  nvvm.store.async.shared %addr, %v128, mbarrier = %mbar : !llvm.ptr<7>, i128, !llvm.ptr<7>
  llvm.return
}
