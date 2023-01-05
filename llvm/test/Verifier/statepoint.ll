; RUN: opt -S %s -passes=verify | FileCheck %s

declare void @use(...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare i32 @"personality_function"()

;; Basic usage
define ptr addrspace(1) @test1(ptr addrspace(1) %arg) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg), "deopt" (i32 0, i32 0, i32 0, i32 10, i32 0)]
  %reloc = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 1)
  ;; It is perfectly legal to relocate the same value multiple times...
  %reloc2 = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 1)
  %reloc3 = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 1, i32 0)
  ret ptr addrspace(1) %reloc
; CHECK-LABEL: test1
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK: gc.relocate
; CHECK: gc.relocate
; CHECK: ret ptr addrspace(1) %reloc
}

; This test catches two cases where the verifier was too strict:
; 1) A base doesn't need to be relocated if it's never used again
; 2) A value can be replaced by one which is known equal.  This
; means a potentially derived pointer can be known base and that
; we can't check that derived pointer are never bases.
define void @test2(ptr addrspace(1) %arg, ptr addrspace(1) %arg2) gc "statepoint-example" {
entry:
  %c = icmp eq ptr addrspace(1) %arg,  %arg2
  br i1 %c, label %equal, label %notequal

notequal:
  ret void

equal:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg, ptr addrspace(1) %arg), "deopt" (i32 0, i32 0, i32 0, i32 10, i32 0)]
  %reloc = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 0)
  call void undef(ptr addrspace(1) %reloc)
  ret void
; CHECK-LABEL: test2
; CHECK-LABEL: equal
; CHECK: statepoint
; CHECK-NEXT: %reloc = call
; CHECK-NEXT: call
; CHECK-NEXT: ret voi
}

; Basic test for invoke statepoints
define ptr addrspace(1) @test3(ptr addrspace(1) %obj, ptr addrspace(1) %obj1) gc "statepoint-example" personality ptr @"personality_function" {
; CHECK-LABEL: test3
entry:
  ; CHECK-LABEL: entry
  ; CHECK: statepoint
  %0 = invoke token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %obj, ptr addrspace(1) %obj1), "deopt" (i32 0, i32 -1, i32 0, i32 0, i32 0)]
          to label %normal_dest unwind label %exceptional_return

normal_dest:
  ; CHECK-LABEL: normal_dest:
  ; CHECK: gc.relocate
  ; CHECK: gc.relocate
  ; CHECK: ret
  %obj.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %0, i32 0, i32 0)
  %obj1.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %0, i32 1, i32 1)
  ret ptr addrspace(1) %obj.relocated

exceptional_return:
  ; CHECK-LABEL: exceptional_return
  ; CHECK: gc.relocate
  ; CHECK: gc.relocate
  %landing_pad = landingpad token
          cleanup
  %obj.relocated1 = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %landing_pad, i32 0, i32 0)
  %obj1.relocated1 = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %landing_pad, i32 1, i32 1)
  ret ptr addrspace(1) %obj1.relocated1
}

; Test for statepoint with sret attribute.
; This should be allowed as long as the wrapped function is not vararg.
%struct = type { i64, i64, i64 }

declare void @fn_sret(ptr sret(%struct))

define void @test_sret() gc "statepoint-example" {
  %x = alloca %struct
  %statepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void (ptr)) @fn_sret, i32 1, i32 0, ptr sret(%struct) %x, i32 0, i32 0)
  ret void
  ; CHECK-LABEL: test_sret
  ; CHECK: alloca
  ; CHECK: statepoint
  ; CHECK-SAME: sret
  ; CHECK: ret
}
