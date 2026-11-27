; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx86 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx86 -verify-machineinstrs | FileCheck %s --check-prefix=NO-REPORT

; NO-REPORT-NOT: phase_type
; NO-REPORT-NOT: .reg .b8
; NO-REPORT: mbarrier.test_wait.shared.b64
; NO-REPORT-NOT: phase_type
; NO-REPORT-NOT: .reg .b8

declare i1 @llvm.nvvm.mbarrier.test.wait.scope.cta.space.cta(ptr addrspace(3), i64)
declare i1 @llvm.nvvm.mbarrier.test.wait.parity.scope.cta.space.cta(ptr addrspace(3), i32)
declare i1 @llvm.nvvm.mbarrier.test.wait.scope.cluster.space.cta(ptr addrspace(3), i64)
declare i1 @llvm.nvvm.mbarrier.test.wait.parity.relaxed.scope.cluster.space.cta(ptr addrspace(3), i32)
declare i1 @llvm.nvvm.mbarrier.test.wait.relaxed.scope.cta.space.cta(ptr addrspace(3), i64)
declare i1 @llvm.nvvm.mbarrier.try.wait.tl.scope.cta.space.cta(ptr addrspace(3), i64, i32)
declare i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.relaxed.scope.cluster.space.cta(ptr addrspace(3), i32, i32)

define i1 @legacy_test_state_cta(ptr addrspace(3) %bar, i64 %state) {
; CHECK-LABEL: legacy_test_state_cta(
; CHECK: mbarrier.test_wait.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}], %{{rd[0-9]+}};
; CHECK: selp.b32 {{.*}}, -1, 0, %[[P]];
  %complete = call i1 @llvm.nvvm.mbarrier.test.wait.scope.cta.space.cta(ptr addrspace(3) %bar, i64 %state)
  ret i1 %complete
}

define i1 @legacy_test_parity_cta(ptr addrspace(3) %bar, i32 %phase) {
; CHECK-LABEL: legacy_test_parity_cta(
; CHECK: mbarrier.test_wait.parity.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}], %{{r[0-9]+}};
; CHECK: selp.b32 {{.*}}, -1, 0, %[[P]];
  %complete = call i1 @llvm.nvvm.mbarrier.test.wait.parity.scope.cta.space.cta(ptr addrspace(3) %bar, i32 %phase)
  ret i1 %complete
}

define i1 @legacy_test_state_cluster(ptr addrspace(3) %bar, i64 %state) {
; CHECK-LABEL: legacy_test_state_cluster(
; CHECK: mbarrier.test_wait.acquire.cluster.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}], %{{rd[0-9]+}};
; CHECK: selp.b32 {{.*}}, -1, 0, %[[P]];
  %complete = call i1 @llvm.nvvm.mbarrier.test.wait.scope.cluster.space.cta(ptr addrspace(3) %bar, i64 %state)
  ret i1 %complete
}

define i1 @legacy_test_parity_cluster_relaxed(ptr addrspace(3) %bar, i32 %phase) {
; CHECK-LABEL: legacy_test_parity_cluster_relaxed(
; CHECK: mbarrier.test_wait.parity.relaxed.cluster.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}], %{{r[0-9]+}};
; CHECK: selp.b32 {{.*}}, -1, 0, %[[P]];
  %complete = call i1 @llvm.nvvm.mbarrier.test.wait.parity.relaxed.scope.cluster.space.cta(ptr addrspace(3) %bar, i32 %phase)
  ret i1 %complete
}

define i1 @legacy_try_state_cta_timeout(ptr addrspace(3) %bar, i64 %state, i32 %ticks) {
; CHECK-LABEL: legacy_try_state_cta_timeout(
; CHECK: mbarrier.try_wait.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}], %{{rd[0-9]+}}, %{{r[0-9]+}};
; CHECK: selp.b32 {{.*}}, -1, 0, %[[P]];
  %complete = call i1 @llvm.nvvm.mbarrier.try.wait.tl.scope.cta.space.cta(ptr addrspace(3) %bar, i64 %state, i32 %ticks)
  ret i1 %complete
}

define i1 @legacy_try_parity_cluster_relaxed_timeout(ptr addrspace(3) %bar, i32 %phase, i32 %ticks) {
; CHECK-LABEL: legacy_try_parity_cluster_relaxed_timeout(
; CHECK: mbarrier.try_wait.parity.relaxed.cluster.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}], %{{r[0-9]+}}, %{{r[0-9]+}};
; CHECK: selp.b32 {{.*}}, -1, 0, %[[P]];
  %complete = call i1 @llvm.nvvm.mbarrier.try.wait.parity.tl.relaxed.scope.cluster.space.cta(ptr addrspace(3) %bar, i32 %phase, i32 %ticks)
  ret i1 %complete
}

define i1 @legacy_offset_pointer(ptr addrspace(3) %bar, i64 %state) {
; CHECK-LABEL: legacy_offset_pointer(
; CHECK: mbarrier.test_wait.relaxed.cta.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}+8], %{{rd[0-9]+}};
; CHECK: selp.b32 {{.*}}, -1, 0, %[[P]];
  %addr = getelementptr i8, ptr addrspace(3) %bar, i32 8
  %complete = call i1 @llvm.nvvm.mbarrier.test.wait.relaxed.scope.cta.space.cta(ptr addrspace(3) %addr, i64 %state)
  ret i1 %complete
}

define i32 @legacy_branch_on_result(ptr addrspace(3) %bar, i32 %phase) {
; CHECK-LABEL: legacy_branch_on_result(
; CHECK: mbarrier.test_wait.parity.shared.b64 %[[P:p[0-9]+]], [%{{rd[0-9]+}}], %{{r[0-9]+}};
; CHECK: not.pred %[[NOTP:p[0-9]+]], %[[P]];
; CHECK: @%[[NOTP]] bra
  %complete = call i1 @llvm.nvvm.mbarrier.test.wait.parity.scope.cta.space.cta(ptr addrspace(3) %bar, i32 %phase)
  br i1 %complete, label %completed, label %pending
completed:
  ret i32 1
pending:
  ret i32 0
}
