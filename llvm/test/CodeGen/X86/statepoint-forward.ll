; RUN: opt -O3 -S < %s | FileCheck --check-prefix=CHECK-OPT %s
; RUN: llc -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK-LLC %s
; These tests are targetted at making sure we don't retain information
; about memory which contains potential gc references across a statepoint.
; They're carefully written to only outlaw forwarding of references. 
; Depending on the collector, forwarding non-reference fields or
; constant null references may be perfectly legal. (If unimplemented.)
; The general structure of these tests is:
; - learn a fact about memory (via an assume)
; - cross a statepoint
; - check the same fact about memory (which we no longer know)

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; If not at a statepoint, we could forward known memory values
; across this call.
declare void @func() readonly

;; Forwarding the value of a pointer load is invalid since it may have
;; changed at the safepoint.  Forwarding a non-gc pointer value would 
;; be valid, but is not currently implemented.
define i1 @test_load_forward(i32 addrspace(1)* addrspace(1)* %p) gc "statepoint-example" {
entry:
  %before = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %p
  %cmp1 = call i1 @f(i32 addrspace(1)* %before)
  call void @llvm.assume(i1 %cmp1)
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* addrspace(1)* %p)]
  %pnew = call i32 addrspace(1)* addrspace(1)* @llvm.experimental.gc.relocate.p1p1i32(token %safepoint_token,  i32 0, i32 0)
  %after = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %pnew
  %cmp2 = call i1 @f(i32 addrspace(1)* %after)
  ret i1 %cmp2

; CHECK-OPT-LABEL: test_load_forward
; CHECK-OPT: ret i1 %cmp2
; CHECK-LLC-LABEL: test_load_forward
; CHECK-LLC: callq f
}

;; Same as above, but forwarding from a store
define i1 @test_store_forward(i32 addrspace(1)* addrspace(1)* %p,
                              i32 addrspace(1)* %v) gc "statepoint-example" {
entry:
  %cmp1 = call i1 @f(i32 addrspace(1)* %v)
  call void @llvm.assume(i1 %cmp1)
  store i32 addrspace(1)* %v, i32 addrspace(1)* addrspace(1)* %p
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* addrspace(1)* %p)]
  %pnew = call i32 addrspace(1)* addrspace(1)* @llvm.experimental.gc.relocate.p1p1i32(token %safepoint_token,  i32 0, i32 0)
  %after = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %pnew
  %cmp2 = call i1 @f(i32 addrspace(1)* %after)
  ret i1 %cmp2

; CHECK-OPT-LABEL: test_store_forward
; CHECK-OPT: ret i1 %cmp2
; CHECK-LLC-LABEL: test_store_forward
; CHECK-LLC: callq f
}

; A predicate on the pointer which is not simply null, but whose value
; would be known unchanged if the pointer value could be forwarded.
; The implementation of such a function could inspect the integral value
; of the pointer and is thus not safe to reuse after a statepoint.
declare i1 @f(i32 addrspace(1)* %v) readnone

; This is a variant of the test_load_forward test which is intended to 
; highlight the fact that a gc pointer can be stored in part of the heap
; that is not itself GC managed.  The GC may have an external mechanism
; to know about and update that value at a safepoint.  Note that the 
; statepoint does not provide the collector with this root.
define i1 @test_load_forward_nongc_heap(i32 addrspace(1)** %p) gc "statepoint-example" {
entry:
  %before = load i32 addrspace(1)*, i32 addrspace(1)** %p
  %cmp1 = call i1 @f(i32 addrspace(1)* %before)
  call void @llvm.assume(i1 %cmp1)
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0)
  %after = load i32 addrspace(1)*, i32 addrspace(1)** %p
  %cmp2 = call i1 @f(i32 addrspace(1)* %after)
  ret i1 %cmp2

; CHECK-OPT-LABEL: test_load_forward_nongc_heap
; CHECK-OPT: ret i1 %cmp2
; CHECK-LLC-LABEL: test_load_forward_nongc_heap
; CHECK-LLC: callq f
}

;; Same as above, but forwarding from a store
define i1 @test_store_forward_nongc_heap(i32 addrspace(1)** %p,
                                         i32 addrspace(1)* %v) gc "statepoint-example" {
entry:
  %cmp1 = call i1 @f(i32 addrspace(1)* %v)
  call void @llvm.assume(i1 %cmp1)
  store i32 addrspace(1)* %v, i32 addrspace(1)** %p
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0)
  %after = load i32 addrspace(1)*, i32 addrspace(1)** %p
  %cmp2 = call i1 @f(i32 addrspace(1)* %after)
  ret i1 %cmp2

; CHECK-OPT-LABEL: test_store_forward_nongc_heap
; CHECK-OPT: ret i1 %cmp2
; CHECK-LLC-LABEL: test_store_forward_nongc_heap
; CHECK-LLC: callq f
}

declare void @llvm.assume(i1)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 addrspace(1)* addrspace(1)* @llvm.experimental.gc.relocate.p1p1i32(token, i32, i32) #3
