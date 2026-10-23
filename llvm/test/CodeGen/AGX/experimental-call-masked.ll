; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; A masked caller enters a divergent callee. A post-RA spill reload must stay
; after the return's mask pop, and the callee must inherit r0l from its caller.
; CHECK-LABEL: kernel:
; CHECK: call leaf
; CHECK-NEXT: pop_exec r0l, 2
; CHECK-NEXT: stack_load
; CHECK-LABEL: leaf:
; CHECK-NOT: mov_imm r0l, 0
; CHECK: if_icmp
; CHECK: ret r1

target triple = "agx-unknown-unknown"
declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @kernel() {
entry:
  %id = call i32 @llvm.agx.thread.position.in.grid.x()
  %cond = icmp ult i32 %id, 16
  br i1 %cond, label %do_call, label %skip
do_call:
  %called = call i32 @leaf(i32 %id)
  br label %merge
skip:
  %other = add i32 %id, 100
  br label %merge
merge:
  %value = phi i32 [%called, %do_call], [%other, %skip]
  %off = shl i32 %id, 2
  %addr = add i32 %off, 8192
  %ptr = inttoptr i32 %addr to ptr addrspace(1)
  store i32 %value, ptr addrspace(1) %ptr
  ret void
}
define internal i32 @leaf(i32 %x) #0 {
entry:
  %islo = icmp ult i32 %x, 8
  br i1 %islo, label %lo, label %hi
lo:
  %a = add i32 %x, 5
  br label %end
hi:
  %b = shl i32 %x, 1
  br label %end
end:
  %v = phi i32 [%a, %lo], [%b, %hi]
  ret i32 %v
}
attributes #0 = { noinline }
