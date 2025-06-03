// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @prefetch_L1(%gen_ptr: !llvm.ptr, %local_ptr: !llvm.ptr<5>, %global_ptr: !llvm.ptr<1>) {
  // CHECK-LABEL: define void @prefetch_L1(ptr %0, ptr addrspace(5) %1, ptr addrspace(1) %2) {
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.L1(ptr %0)
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.local.L1(ptr addrspace(5) %1)
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.global.L1(ptr addrspace(1) %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.prefetch level = L1, %gen_ptr : !llvm.ptr<0>
  nvvm.prefetch level = L1, %local_ptr : !llvm.ptr<5>
  nvvm.prefetch level = L1, %global_ptr : !llvm.ptr<1>
  llvm.return
}

llvm.func @prefetch_L2(%gen_ptr: !llvm.ptr, %local_ptr: !llvm.ptr<5>, %global_ptr: !llvm.ptr<1>) {
  // CHECK-LABEL: define void @prefetch_L2(ptr %0, ptr addrspace(5) %1, ptr addrspace(1) %2) {
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.L2(ptr %0)
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.local.L2(ptr addrspace(5) %1)
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.global.L2(ptr addrspace(1) %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.prefetch level = L2, %gen_ptr : !llvm.ptr<0>
  nvvm.prefetch level = L2, %local_ptr : !llvm.ptr<5>
  nvvm.prefetch level = L2, %global_ptr : !llvm.ptr<1>
  llvm.return
}

llvm.func @prefetch_L2_eviction_priority(%global_ptr: !llvm.ptr<1>) {
  // CHECK-LABEL: define void @prefetch_L2_eviction_priority(ptr addrspace(1) %0) {
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.global.L2.evict.last(ptr addrspace(1) %0)
  // CHECK-NEXT: call void @llvm.nvvm.prefetch.global.L2.evict.normal(ptr addrspace(1) %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.prefetch level = L2, %global_ptr, evict_priority = evict_last : !llvm.ptr<1>
  nvvm.prefetch level = L2, %global_ptr, evict_priority = evict_normal : !llvm.ptr<1>
  llvm.return
}

llvm.func @prefetch_L1_uniform(%gen_ptr: !llvm.ptr) {
  // CHECK-LABEL: define void @prefetch_L1_uniform(ptr %0) {
  // CHECK-NEXT: call void @llvm.nvvm.prefetchu.L1(ptr %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.prefetch level = L1 uniform, %gen_ptr : !llvm.ptr
  llvm.return
}
