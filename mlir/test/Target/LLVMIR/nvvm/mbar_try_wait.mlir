// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_try_wait_state(%barrier: !llvm.ptr, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_try_wait_state(ptr %0, i64 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.scope.cta.space.cta(ptr addrspace(3) %3, i64 %1)
  // CHECK-NEXT: %5 = extractvalue { i1, i1, i32 } %4, 0
  // CHECK-NEXT: %6 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.scope.cluster.space.cta(ptr addrspace(3) %6, i64 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %9, i64 %1)
  // CHECK-NEXT: %11 = extractvalue { i1, i1, i32 } %10, 0
  // CHECK-NEXT: %12 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %13 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %12, i64 %1)
  // CHECK-NEXT: %14 = extractvalue { i1, i1, i32 } %13, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state : !llvm.ptr, i64 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state  scope = cluster : !llvm.ptr, i64 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state  relaxed = true : !llvm.ptr, i64 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state  scope = cluster relaxed = true : !llvm.ptr, i64 -> i1

  llvm.return
}

llvm.func @mbarrier_try_wait_state_with_timelimit(%barrier: !llvm.ptr, %state : i64, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_state_with_timelimit(ptr %0, i64 %1, i32 %2) {
  // CHECK-NEXT: %4 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %5 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.scope.cta.space.cta(ptr addrspace(3) %4, i64 %1, i32 %2)
  // CHECK-NEXT: %6 = extractvalue { i1, i1, i32 } %5, 0
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.scope.cluster.space.cta(ptr addrspace(3) %7, i64 %1, i32 %2)
  // CHECK-NEXT: %9 = extractvalue { i1, i1, i32 } %8, 0
  // CHECK-NEXT: %10 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %11 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %10, i64 %1, i32 %2)
  // CHECK-NEXT: %12 = extractvalue { i1, i1, i32 } %11, 0
  // CHECK-NEXT: %13 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %14 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %13, i64 %1, i32 %2)
  // CHECK-NEXT: %15 = extractvalue { i1, i1, i32 } %14, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state, %ticks : !llvm.ptr, i64, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state, %ticks  scope = cluster : !llvm.ptr, i64, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state, %ticks  relaxed = true : !llvm.ptr, i64, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state, %ticks  scope = cluster relaxed = true : !llvm.ptr, i64, i32 -> i1

  llvm.return
}

llvm.func @mbarrier_try_wait_shared_state(%barrier: !llvm.ptr<3>, %state : i64) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_state(ptr addrspace(3) %0, i64 %1) {
  // CHECK-NEXT: %3 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %4 = extractvalue { i1, i1, i32 } %3, 0
  // CHECK-NEXT: %5 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %6 = extractvalue { i1, i1, i32 } %5, 0
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1)
  // CHECK-NEXT: %10 = extractvalue { i1, i1, i32 } %9, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state : !llvm.ptr<3>, i64 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state  scope = cluster : !llvm.ptr<3>, i64 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state  relaxed = true : !llvm.ptr<3>, i64 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state  scope = cluster relaxed = true : !llvm.ptr<3>, i64 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_shared_state_with_timelimit(%barrier: !llvm.ptr<3>, %state : i64, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_state_with_timelimit(ptr addrspace(3) %0, i64 %1, i32 %2) {
  // CHECK-NEXT: %4 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: %5 = extractvalue { i1, i1, i32 } %4, 0
  // CHECK-NEXT: %6 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: %7 = extractvalue { i1, i1, i32 } %6, 0
  // CHECK-NEXT: %8 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: %9 = extractvalue { i1, i1, i32 } %8, 0
  // CHECK-NEXT: %10 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.tl.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i64 %1, i32 %2)
  // CHECK-NEXT: %11 = extractvalue { i1, i1, i32 } %10, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %state, %ticks : !llvm.ptr<3>, i64, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %state, %ticks  scope = cluster : !llvm.ptr<3>, i64, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %state, %ticks  relaxed = true : !llvm.ptr<3>, i64, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %state, %ticks  scope = cluster relaxed = true : !llvm.ptr<3>, i64, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_phase(%barrier: !llvm.ptr, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_phase(ptr %0, i32 %1) {
  // CHECK-NEXT: %3 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %4 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.scope.cta.space.cta(ptr addrspace(3) %3, i32 %1)
  // CHECK-NEXT: %5 = extractvalue { i1, i1, i32 } %4, 0
  // CHECK-NEXT: %6 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.scope.cluster.space.cta(ptr addrspace(3) %6, i32 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %10 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %9, i32 %1)
  // CHECK-NEXT: %11 = extractvalue { i1, i1, i32 } %10, 0
  // CHECK-NEXT: %12 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %13 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %12, i32 %1)
  // CHECK-NEXT: %14 = extractvalue { i1, i1, i32 } %13, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase : !llvm.ptr, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase  scope = cluster : !llvm.ptr, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase  relaxed = true : !llvm.ptr, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase  scope = cluster relaxed = true : !llvm.ptr, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_phase_with_timelimit(%barrier: !llvm.ptr, %phase : i32, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_phase_with_timelimit(ptr %0, i32 %1, i32 %2) {
  // CHECK-NEXT: %4 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %5 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.scope.cta.space.cta(ptr addrspace(3) %4, i32 %1, i32 %2)
  // CHECK-NEXT: %6 = extractvalue { i1, i1, i32 } %5, 0
  // CHECK-NEXT: %7 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %8 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.scope.cluster.space.cta(ptr addrspace(3) %7, i32 %1, i32 %2)
  // CHECK-NEXT: %9 = extractvalue { i1, i1, i32 } %8, 0
  // CHECK-NEXT: %10 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %11 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %10, i32 %1, i32 %2)
  // CHECK-NEXT: %12 = extractvalue { i1, i1, i32 } %11, 0
  // CHECK-NEXT: %13 = addrspacecast ptr %0 to ptr addrspace(3)
  // CHECK-NEXT: %14 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %13, i32 %1, i32 %2)
  // CHECK-NEXT: %15 = extractvalue { i1, i1, i32 } %14, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks : !llvm.ptr, i32, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks  scope = cluster : !llvm.ptr, i32, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks  relaxed = true : !llvm.ptr, i32, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks  scope = cluster relaxed = true : !llvm.ptr, i32, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_shared_phase(%barrier: !llvm.ptr<3>, %phase : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_phase(ptr addrspace(3) %0, i32 %1) {
  // CHECK-NEXT: %3 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %4 = extractvalue { i1, i1, i32 } %3, 0
  // CHECK-NEXT: %5 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %6 = extractvalue { i1, i1, i32 } %5, 0
  // CHECK-NEXT: %7 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %8 = extractvalue { i1, i1, i32 } %7, 0
  // CHECK-NEXT: %9 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1)
  // CHECK-NEXT: %10 = extractvalue { i1, i1, i32 } %9, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase : !llvm.ptr<3>, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase  scope = cluster : !llvm.ptr<3>, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase  relaxed = true : !llvm.ptr<3>, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase  scope = cluster relaxed = true : !llvm.ptr<3>, i32 -> i1
  llvm.return
}

llvm.func @mbarrier_try_wait_shared_phase_with_timelimit(%barrier: !llvm.ptr<3>, %phase : i32, %ticks : i32) {
  // CHECK-LABEL: define void @mbarrier_try_wait_shared_phase_with_timelimit(ptr addrspace(3) %0, i32 %1, i32 %2) {
  // CHECK-NEXT: %4 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: %5 = extractvalue { i1, i1, i32 } %4, 0
  // CHECK-NEXT: %6 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: %7 = extractvalue { i1, i1, i32 } %6, 0
  // CHECK-NEXT: %8 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.relaxed.scope.cta.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: %9 = extractvalue { i1, i1, i32 } %8, 0
  // CHECK-NEXT: %10 = call { i1, i1, i32 } @llvm.nvvm.mbarrier.try.wait.parity.tl.primary.relaxed.scope.cluster.space.cta(ptr addrspace(3) %0, i32 %1, i32 %2)
  // CHECK-NEXT: %11 = extractvalue { i1, i1, i32 } %10, 0
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  %0 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks : !llvm.ptr<3>, i32, i32 -> i1
  %1 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks  scope = cluster : !llvm.ptr<3>, i32, i32 -> i1

  %2 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks  relaxed = true : !llvm.ptr<3>, i32, i32 -> i1
  %3 = nvvm.mbarrier.try_wait %barrier, %phase, %ticks  scope = cluster relaxed = true : !llvm.ptr<3>, i32, i32 -> i1
  llvm.return
}
