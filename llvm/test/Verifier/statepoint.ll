; RUN: opt -S %s -verify | FileCheck %s

declare void @use(...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32, i32, i32)
declare i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(i32, i32, i32)
declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 @"personality_function"()

;; Basic usage
define i64 addrspace(1)* @test1(i8 addrspace(1)* %arg) gc "statepoint-example" {
entry:
  %cast = bitcast i8 addrspace(1)* %arg to i64 addrspace(1)*
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg, i8 addrspace(1)* %arg)
  %reloc = call i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(i32 %safepoint_token, i32 12, i32 13)
  ;; It is perfectly legal to relocate the same value multiple times...
  %reloc2 = call i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(i32 %safepoint_token, i32 12, i32 13)
  %reloc3 = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 %safepoint_token, i32 13, i32 12)
  ret i64 addrspace(1)* %reloc
; CHECK-LABEL: test1
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK: gc.relocate
; CHECK: gc.relocate
; CHECK: ret i64 addrspace(1)* %reloc
}

; This test catches two cases where the verifier was too strict:
; 1) A base doesn't need to be relocated if it's never used again
; 2) A value can be replaced by one which is known equal.  This
; means a potentially derived pointer can be known base and that
; we can't check that derived pointer are never bases.
define void @test2(i8 addrspace(1)* %arg, i64 addrspace(1)* %arg2) gc "statepoint-example" {
entry:
  %cast = bitcast i8 addrspace(1)* %arg to i64 addrspace(1)*
  %c = icmp eq i64 addrspace(1)* %cast,  %arg2
  br i1 %c, label %equal, label %notequal

notequal:
  ret void

equal:
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg, i8 addrspace(1)* %arg)
  %reloc = call i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(i32 %safepoint_token, i32 12, i32 13)
  call void undef(i64 addrspace(1)* %reloc)
  ret void
; CHECK-LABEL: test2
; CHECK-LABEL: equal
; CHECK: statepoint
; CHECK-NEXT: %reloc = call 
; CHECK-NEXT: call
; CHECK-NEXT: ret voi
}

; Basic test for invoke statepoints
define i8 addrspace(1)* @test3(i8 addrspace(1)* %obj, i8 addrspace(1)* %obj1) gc "statepoint-example" personality i32 ()* @"personality_function" {
; CHECK-LABEL: test3
entry:
  ; CHECK-LABEL: entry
  ; CHECK: statepoint
  %0 = invoke i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i8 addrspace(1)* %obj, i8 addrspace(1)* %obj1)
          to label %normal_dest unwind label %exceptional_return

normal_dest:
  ; CHECK-LABEL: normal_dest:
  ; CHECK: gc.relocate
  ; CHECK: gc.relocate
  ; CHECK: ret
  %obj.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 %0, i32 12, i32 12)
  %obj1.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 %0, i32 12, i32 12)
  ret i8 addrspace(1)* %obj.relocated

exceptional_return:
  ; CHECK-LABEL: exceptional_return
  ; CHECK: gc.relocate
  ; CHECK: gc.relocate
  %landing_pad = landingpad { i8*, i32 }
          cleanup
  %relocate_token = extractvalue { i8*, i32 } %landing_pad, 1
  %obj.relocated1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 %relocate_token, i32 12, i32 12)
  %obj1.relocated1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 %relocate_token, i32 12, i32 12)
  ret i8 addrspace(1)* %obj1.relocated1
}
