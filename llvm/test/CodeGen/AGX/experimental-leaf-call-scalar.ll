; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
;
; The argument and result use r2. The original load remains live across the
; call, so register allocation must spill and reload it. A wait must complete
; the asynchronous load before the callee reads r2.
;
; CHECK-LABEL: kernel:
; CHECK: device_load
; CHECK: wait 0
; CHECK: stack_store
; CHECK: if_fcmp r0l, eq, 0.0, 0.0, 2
; CHECK: call leaf
; CHECK-NEXT: pop_exec r0l, 2
; CHECK: stack_load
; CHECK-LABEL: leaf:
; CHECK: iadd r2, r2, 5
; CHECK-NEXT: ret r1

define agx_kernel void @kernel() {
entry:
  %v = load volatile i32, ptr addrspace(1) inttoptr (i64 8192 to ptr addrspace(1))
  %w = call i32 @leaf(i32 %v)
  %sum = add i32 %v, %w
  store volatile i32 %sum, ptr addrspace(1) inttoptr (i64 8196 to ptr addrspace(1))
  ret void
}

define internal i32 @leaf(i32 %x) #0 {
entry:
  %sum = add i32 %x, 5
  ret i32 %sum
}

attributes #0 = { noinline }
