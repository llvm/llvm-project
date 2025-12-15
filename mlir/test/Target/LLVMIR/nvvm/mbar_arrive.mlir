// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_arrive_generic(%barrier: !llvm.ptr, %count : i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_generic(ptr %0, i32 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %3, i32 1)
  // CHECK-NEXT: %5 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %6 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %5, i32 %1)
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %7, i32 %1)
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cluster.space.cta(ptr addrspace(3) %9, i32 %1)
  // CHECK-NEXT: %11 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %12 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cta(ptr addrspace(3) %11, i32 1)
  // CHECK-NEXT: %13 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %14 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cta(ptr addrspace(3) %13, i32 %1)
  // CHECK-NEXT: %15 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %16 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cta(ptr addrspace(3) %15, i32 %1)
  // CHECK-NEXT: %17 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %18 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cluster.space.cta(ptr addrspace(3) %17, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.arrive %barrier : !llvm.ptr  -> i64
  %1 = nvvm.mbarrier.arrive %barrier, %count : !llvm.ptr  -> i64
  %2 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cta>} : !llvm.ptr  -> i64
  %3 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr  -> i64

  %4 = nvvm.mbarrier.arrive %barrier {relaxed = true} : !llvm.ptr  -> i64
  %5 = nvvm.mbarrier.arrive %barrier, %count {relaxed = true} : !llvm.ptr  -> i64
  %6 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cta>, relaxed = true} : !llvm.ptr  -> i64
  %7 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cluster>, relaxed = true} : !llvm.ptr  -> i64
  llvm.return
}

llvm.func @mbarrier_arrive_shared(%barrier: !llvm.ptr<3>, %count : i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_shared(ptr addrspace(3) %0, i32 %1) {
  // CHECK-NEXT: %3 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %0, i32 1)
  // CHECK-NEXT: %4 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %5 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %6 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %7 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 1)
  // CHECK-NEXT: %8 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %9 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %10 = call i64 @llvm.nvvm.mbarrier.arrive.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.arrive %barrier : !llvm.ptr<3>  -> i64
  %1 = nvvm.mbarrier.arrive %barrier, %count : !llvm.ptr<3>  -> i64
  %2 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<3>  -> i64
  %3 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>  -> i64

  %4 = nvvm.mbarrier.arrive %barrier {relaxed = true} : !llvm.ptr<3>  -> i64
  %5 = nvvm.mbarrier.arrive %barrier, %count {relaxed = true} : !llvm.ptr<3>  -> i64
  %6 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cta>, relaxed = true} : !llvm.ptr<3>  -> i64
  %7 = nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cluster>, relaxed = true} : !llvm.ptr<3>  -> i64
  llvm.return
}

llvm.func @mbarrier_arrive_shared_cluster(%barrier: !llvm.ptr<7>, %count : i32) {
  // CHECK-LABEL: define void @mbarrier_arrive_shared_cluster(ptr addrspace(7) %0, i32 %1) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.scope.cta.space.cluster(ptr addrspace(7) %0, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.scope.cluster.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cluster(ptr addrspace(7) %0, i32 1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.relaxed.scope.cta.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.arrive.relaxed.scope.cluster.space.cluster(ptr addrspace(7) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.arrive %barrier : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cta>} : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<7>

  nvvm.mbarrier.arrive %barrier {relaxed = true} : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count {relaxed = true} : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cta>, relaxed = true} : !llvm.ptr<7>
  nvvm.mbarrier.arrive %barrier, %count {scope = #nvvm.mem_scope<cluster>, relaxed = true} : !llvm.ptr<7>
  llvm.return
}

llvm.func @mbarrier_arrive_nocomplete(%barrier: !llvm.ptr) {
  // CHECK-LABEL: define void @mbarrier_arrive_nocomplete(ptr %0) {
  // CHECK-NEXT: %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-NEXT: %3 = call i64 @llvm.nvvm.mbarrier.arrive.noComplete(ptr %0, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  %0 = nvvm.mbarrier.arrive.nocomplete %barrier, %count : !llvm.ptr, i32 -> i64
  llvm.return
}

llvm.func @mbarrier_arrive_nocomplete_shared(%barrier: !llvm.ptr<3>) {
  // CHECK-LABEL: define void @mbarrier_arrive_nocomplete_shared(ptr addrspace(3) %0) {
  // CHECK-NEXT: %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-NEXT: %3 = call i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared(ptr addrspace(3) %0, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  %0 = nvvm.mbarrier.arrive.nocomplete %barrier, %count : !llvm.ptr<3>, i32  -> i64
  llvm.return
}

llvm.func @mbarrier_arrive_ignore_retval(%count : i32, %barrier: !llvm.ptr<3>) {
  // CHECK-LABEL: define void @mbarrier_arrive_ignore_retval(i32 %0, ptr addrspace(3) %1) {
  // CHECK-NEXT: %3 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %1, i32 %0)
  // CHECK-NEXT: %4 = call i64 @llvm.nvvm.mbarrier.arrive.scope.cta.space.cta(ptr addrspace(3) %1, i32 %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.arrive %barrier, %count : !llvm.ptr<3>
  nvvm.mbarrier.arrive %barrier, %count : !llvm.ptr<3>

  llvm.return
}

