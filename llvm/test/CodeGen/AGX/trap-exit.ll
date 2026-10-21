; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; An executed trap does nothing on G14C (ptx-re-blockers B21), so a thread
; that reaches llvm.trap and unreachable (as in __assertfail) returns there:
; the code after it does not run.

declare i32 @llvm.agx.thread.position.in.grid.x()
declare void @llvm.trap()

; The trapping threads skip the store.
; CHECK-LABEL: divergent:
; CHECK:         if_icmp r0l, ult
; CHECK:         device_store
; CHECK:         else_fcmp
; CHECK-NEXT:    jmp_exec_none [[RET:.LBB[0-9_]+]]
; CHECK:         trap
; CHECK-NEXT:  [[RET]]:
; CHECK-NEXT:    pop_exec
; CHECK-NEXT:    stop
define agx_kernel void @divergent(ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %bad = icmp uge i32 %i, %n
  br i1 %bad, label %fail, label %ok
fail:
  call void @llvm.trap()
  unreachable
ok:
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 7, ptr addrspace(1) %p, align 4
  ret void
}

; A kernel whose only exit is unreachable still stops.
; CHECK-LABEL: always:
; CHECK:         trap
; CHECK-NEXT:    stop
define agx_kernel void @always(ptr addrspace(1) %out) {
  call void @llvm.trap()
  unreachable
}
