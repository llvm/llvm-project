;; Test that AMDGPU targets generate uniform counter instrumentation for
;; divergence tracking. This enables PGO to detect which blocks execute
;; uniformly (all lanes active) vs divergently (partial wave execution).

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_test123 = addrspace(1) global i8 0
@__profn_test_kernel = private constant [11 x i8] c"test_kernel"

define amdgpu_kernel void @test_kernel() {
  call void @llvm.instrprof.increment(ptr @__profn_test_kernel, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; Check that uniform counter array is created
; CHECK: @__profu_all_test123 = protected addrspace(1) global

;; Check that ballot intrinsic is used to get active mask
; CHECK: call i32 @llvm.amdgcn.ballot.i32(i1 true)

;; Check that ctpop is used to count active lanes
; CHECK: call i32 @llvm.ctpop.i32

;; Check that uniformity check compares active mask to full wave mask (0xFFFFFFFF)
; CHECK: icmp eq i32 %{{.*}}, -1

;; Check that uniform counter is conditionally updated based on uniformity
;; The atomic wave leader mode uses a branch on isUniform
; CHECK: br i1 %isUniform, label %uniform_then
