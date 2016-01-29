; RUN: opt -rewrite-statepoints-for-gc -S < %s | FileCheck %s

; Ensure that the gc.statepoint calls / invokes we generate have the
; set of arguments we expect it to have.

define i64 addrspace(1)* @test_invoke_format(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1) gc "statepoint-example" personality i32 ()* @personality {
; CHECK-LABEL: @test_invoke_format(
; CHECK-LABEL: entry:
; CHECK: invoke token (i64, i32, i64 addrspace(1)* (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64p1i64f(i64 2882400000, i32 0, i64 addrspace(1)* (i64 addrspace(1)*)* @callee, i32 1, i32 0, i64 addrspace(1)* %obj, i32 0, i32 0, i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1)
entry:
  %ret_val = invoke i64 addrspace(1)* @callee(i64 addrspace(1)* %obj)
               to label %normal_return unwind label %exceptional_return

normal_return:
  ret i64 addrspace(1)* %ret_val

exceptional_return:
  %landing_pad4 = landingpad token
          cleanup
  ret i64 addrspace(1)* %obj1
}

define i64 addrspace(1)* @test_call_format(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1) gc "statepoint-example" {
; CHECK-LABEL: @test_call_format(
; CHECK-LABEL: entry:
; CHECK: call token (i64, i32, i64 addrspace(1)* (i64 addrspace(1)*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p1i64p1i64f(i64 2882400000, i32 0, i64 addrspace(1)* (i64 addrspace(1)*)* @callee, i32 1, i32 0, i64 addrspace(1)* %obj, i32 0, i32 0, i64 addrspace(1)* %obj)
entry:
  %ret_val = call i64 addrspace(1)* @callee(i64 addrspace(1)* %obj)
  ret i64 addrspace(1)* %ret_val
}

; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}

declare i64 addrspace(1)* @callee(i64 addrspace(1)*)
declare i32 @personality()
