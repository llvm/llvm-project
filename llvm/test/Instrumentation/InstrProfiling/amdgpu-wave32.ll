;; Test that AMDGPU PGO instrumentation correctly handles Wave32 targets.
;; Wave32 (e.g., gfx1100) should use:
;; - ballot.i32
;; - mbcnt.lo only (no mbcnt.hi) for lane ID
;; - i32 types for cttz and ctpop on the ballot mask
;; - Full wave mask of 0xFFFFFFFF for uniformity check

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_abcdef456 = addrspace(1) global i8 0
@__profn_kernel_w32 = private constant [10 x i8] c"kernel_w32"

;; The function has target-cpu=gfx1100 which defaults to Wave32
define amdgpu_kernel void @kernel_w32() #0 {
  call void @llvm.instrprof.increment(ptr @__profn_kernel_w32, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

attributes #0 = { "target-cpu"="gfx1100" }

;; Check Wave32: mbcnt.lo only, no mbcnt.hi
; CHECK: %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK-NOT: mbcnt.hi
; CHECK: %lane = and i32 %mbcnt.lo, 31

;; Check that ballot.i32 is used
; CHECK: %activeMask = call i32 @llvm.amdgcn.ballot.i32(i1 true)

;; Check that cttz.i32 is used for leader election
; CHECK: call i32 @llvm.cttz.i32(i32 %activeMask, i1 true)

;; Check that ctpop.i32 is used for active lane count
; CHECK: call i32 @llvm.ctpop.i32(i32 %activeMask)

;; Check that uniformity check uses 32-bit mask (-1 in i32)
; CHECK: %isUniform = icmp eq i32 %activeMask, -1
