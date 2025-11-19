// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @cp_async_mbarrier_arrive(%bar_shared: !llvm.ptr<3>, %bar_gen: !llvm.ptr) {
  // CHECK-LABEL: define void @cp_async_mbarrier_arrive(ptr addrspace(3) %0, ptr %1) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.mbarrier.arrive(ptr %1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc(ptr %1)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.mbarrier.arrive.shared(ptr addrspace(3) %0)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(ptr addrspace(3) %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.mbarrier.arrive %bar_gen : !llvm.ptr
  nvvm.cp.async.mbarrier.arrive %bar_gen {noinc = true} : !llvm.ptr
  nvvm.cp.async.mbarrier.arrive %bar_shared : !llvm.ptr<3>
  nvvm.cp.async.mbarrier.arrive %bar_shared {noinc = true} : !llvm.ptr<3>
  llvm.return
}

llvm.func @mbarrier_init_generic(%barrier: !llvm.ptr) {
  // CHECK-LABEL: define void @mbarrier_init_generic(ptr %0) {
  // CHECK-NEXT: %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.init(ptr %0, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  nvvm.mbarrier.init %barrier, %count : !llvm.ptr, i32
  llvm.return
}

llvm.func @mbarrier_init_shared(%barrier: !llvm.ptr<3>) {
  // CHECK-LABEL: define void @mbarrier_init_shared(ptr addrspace(3) %0) {
  // CHECK-NEXT: %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) %0, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  nvvm.mbarrier.init %barrier, %count : !llvm.ptr<3>, i32
  llvm.return
}

llvm.func @mbarrier_inval_generic(%barrier: !llvm.ptr) {
  // CHECK-LABEL: define void @mbarrier_inval_generic(ptr %0) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.inval(ptr %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.inval %barrier : !llvm.ptr
  llvm.return
}

llvm.func @mbarrier_inval_shared(%barrier: !llvm.ptr<3>) {
  // CHECK-LABEL: define void @mbarrier_inval_shared(ptr addrspace(3) %0) {
  // CHECK-NEXT: call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.mbarrier.inval %barrier : !llvm.ptr<3>
  llvm.return
}

llvm.func @mbarrier_arrive(%barrier: !llvm.ptr) {
  // CHECK-LABEL: define void @mbarrier_arrive(ptr %0) {
  // CHECK-NEXT: %2 = call i64 @llvm.nvvm.mbarrier.arrive(ptr %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.arrive %barrier : !llvm.ptr  -> i64
  llvm.return
}

llvm.func @mbarrier_arrive_shared(%barrier: !llvm.ptr<3>) {
  // CHECK-LABEL: define void @mbarrier_arrive_shared(ptr addrspace(3) %0) {
  // CHECK-NEXT: %2 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %0)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.arrive %barrier : !llvm.ptr<3> -> i64
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

llvm.func @mbarrier_test_wait(%barrier: !llvm.ptr, %token : i64) -> i1 {
  // CHECK-LABEL: define i1 @mbarrier_test_wait(ptr %0, i64 %1) {
  // CHECK-NEXT: %3 = call i1 @llvm.nvvm.mbarrier.test.wait(ptr %0, i64 %1)
  // CHECK-NEXT: ret i1 %3
  // CHECK-NEXT: }
  %isComplete = nvvm.mbarrier.test.wait %barrier, %token : !llvm.ptr, i64 -> i1
  llvm.return %isComplete : i1
}

llvm.func @mbarrier_test_wait_shared(%barrier: !llvm.ptr<3>, %token : i64) {
  // CHECK-LABEL: define void @mbarrier_test_wait_shared(ptr addrspace(3) %0, i64 %1) {
  // CHECK-NEXT: %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.test.wait.shared(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %count = nvvm.read.ptx.sreg.ntid.x : i32
  %isComplete = nvvm.mbarrier.test.wait %barrier, %token : !llvm.ptr<3>, i64 -> i1
  llvm.return
}
