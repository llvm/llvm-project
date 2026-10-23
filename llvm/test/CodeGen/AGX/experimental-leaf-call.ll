; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
;
; This narrow opt-in path calls a void, argument-free leaf from the kernel's
; entry block. The return needs pop_exec 2 before subsequent kernel work.
;
; CHECK-LABEL: kernel:
; CHECK: if_fcmp r0l, eq, 0.0, 0.0, 2
; CHECK-NEXT: call leaf
; CHECK-NEXT: pop_exec r0l, 2
; CHECK-NEXT: stop
; CHECK-LABEL: leaf:
; CHECK: ret r1

define agx_kernel void @kernel() {
entry:
  call void @leaf()
  ret void
}

define internal void @leaf() #0 {
entry:
  store volatile i32 7, ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1))
  ret void
}

attributes #0 = { noinline }
