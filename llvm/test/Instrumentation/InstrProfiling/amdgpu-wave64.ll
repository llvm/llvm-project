;; Test that AMDGPU PGO instrumentation correctly handles Wave64 targets.
;; Wave64 (e.g., gfx908) should use:
;; - ballot.i64 instead of ballot.i32
;; - mbcnt.lo + mbcnt.hi for lane ID across 64 lanes
;; - i64 types for cttz and ctpop on the ballot mask
;; - Full wave mask of 0xFFFFFFFFFFFFFFFF for uniformity check

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_abcdef123 = addrspace(1) global i8 0
@__profn_kernel_w64 = private constant [10 x i8] c"kernel_w64"

;; The function has target-cpu=gfx908 which defaults to Wave64
define amdgpu_kernel void @kernel_w64() #0 {
  call void @llvm.instrprof.increment(ptr @__profn_kernel_w64, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

attributes #0 = { "target-cpu"="gfx908" }

;; Check that mbcnt.hi is used for Wave64 lane ID computation
; CHECK: %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK: %mbcnt.hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbcnt.lo)
; CHECK: %lane = and i32 %mbcnt.hi, 63

;; Check that ballot.i64 is used instead of ballot.i32
; CHECK: %activeMask = call i64 @llvm.amdgcn.ballot.i64(i1 true)

;; Check that cttz.i64 is used for leader election
; CHECK: call i64 @llvm.cttz.i64(i64 %activeMask, i1 true)

;; Check that ctpop.i64 is used for active lane count
; CHECK: call i64 @llvm.ctpop.i64(i64 %activeMask)

;; Check that uniformity check uses full 64-bit mask (-1 in i64)
; CHECK: %isUniform = icmp eq i64 %activeMask, -1
