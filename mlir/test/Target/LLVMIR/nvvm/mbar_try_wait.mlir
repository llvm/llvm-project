// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_try_wait_state(%barrier: !llvm.ptr, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_try_wait_state(ptr %0, i64 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta(ptr addrspace(3) %3, i64 %1)
  // CHECK-NEXT: %5 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.try.wait.scope.cluster.space.cta(ptr addrspace(3) %5, i64 %1)
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call i1 @llvm.nvvm.mbarrier.try.wait.relaxed.scope.cta.space.cta(ptr addrspace(3) %7, i64 %1)
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call i1 @llvm.nvvm.mbarrier.try.wait.relaxed.scope.cluster.space.cta(ptr addrspace(3) %9, i64 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state : !llvm.ptr, i64 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i64 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state {relaxed = true} : !llvm.ptr, i64 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i64 -> i1

  llvm.return
}

llvm.func @mbarrier_try_wait_state_with_timelimit(%barrier: !llvm.ptr, %state : i64, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_state_with_timelimit(ptr %0, i64 %1, i32 %2) {
  // CHECK-NEXT: %4 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.scope.cta.space.cta(ptr addrspace(3) %4, i64 %1, i32 %2)
  // CHECK-NEXT: %6 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %7 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.scope.cluster.space.cta(ptr addrspace(3) %6, i64 %1, i32 %2)
  // CHECK-NEXT: %8 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %9 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.relaxed.scope.cta.space.cta(ptr addrspace(3) %8, i64 %1, i32 %2)
  // CHECK-NEXT: %10 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %11 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.relaxed.scope.cluster.space.cta(ptr addrspace(3) %10, i64 %1, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state, %ticks : !llvm.ptr, i64, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state, %ticks {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i64, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state, %ticks {relaxed = true} : !llvm.ptr, i64, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state, %ticks {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i64, i32 -> i1

  llvm.return
}

llvm.func @mbarrier_try_wait_shared_state(%barrier: !llvm.ptr<3>, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_state(ptr addrspace(3) %0, i64 %1) {
  // CHECK-NEXT: %3 = call i1 @llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.try.wait.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.try.wait.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.try.wait.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state : !llvm.ptr<3>, i64 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i64 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state {relaxed = true} : !llvm.ptr<3>, i64 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i64 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_shared_state_with_timelimit(%barrier: !llvm.ptr<3>, %state : i64, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_state_with_timelimit(ptr addrspace(3) %0, i64 %1, i32 %2) {
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: %7 = call i1 @llvm.nvvm.mbarrier.try.wait.tl.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state, %ticks : !llvm.ptr<3>, i64, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state, %ticks {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i64, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state, %ticks {relaxed = true} : !llvm.ptr<3>, i64, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state, %ticks {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i64, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_phase(%barrier: !llvm.ptr, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_phase(ptr %0, i32 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.scope.cta.space.cta(ptr addrspace(3) %3, i32 %1)
  // CHECK-NEXT: %5 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.scope.cluster.space.cta(ptr addrspace(3) %5, i32 %1)
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.relaxed.scope.cta.space.cta(ptr addrspace(3) %7, i32 %1)
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.relaxed.scope.cluster.space.cta(ptr addrspace(3) %9, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase : !llvm.ptr, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase {relaxed = true} : !llvm.ptr, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_phase_with_timelimit(%barrier: !llvm.ptr, %phase : i32, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_phase_with_timelimit(ptr %0, i32 %1, i32 %2) {
  // CHECK-NEXT: %4 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.scope.cta.space.cta(ptr addrspace(3) %4, i32 %1, i32 %2)
  // CHECK-NEXT: %6 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %7 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.scope.cluster.space.cta(ptr addrspace(3) %6, i32 %1, i32 %2)
  // CHECK-NEXT: %8 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %9 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.relaxed.scope.cta.space.cta(ptr addrspace(3) %8, i32 %1, i32 %2)
  // CHECK-NEXT: %10 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %11 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.relaxed.scope.cluster.space.cta(ptr addrspace(3) %10, i32 %1, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks : !llvm.ptr, i32, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i32, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks {relaxed = true} : !llvm.ptr, i32, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr, i32, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_shared_phase(%barrier: !llvm.ptr<3>, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_phase(ptr addrspace(3) %0, i32 %1) {
  // CHECK-NEXT: %3 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase : !llvm.ptr<3>, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase {relaxed = true} : !llvm.ptr<3>, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_shared_phase_with_timelimit(%barrier: !llvm.ptr<3>, %phase : i32, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_phase_with_timelimit(ptr addrspace(3) %0, i32 %1, i32 %2) {
  // CHECK-NEXT: %4 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: %5 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: %6 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: %7 = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks : !llvm.ptr<3>, i32, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks {scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks {relaxed = true} : !llvm.ptr<3>, i32, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks {relaxed = true, scope = #nvvm.mem_scope<cluster>} : !llvm.ptr<3>, i32, i32 -> i1
  llvm.return
}
