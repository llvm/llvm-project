; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; Three disjoint stack frames and a saved link permit a direct nested call.
; CHECK-LABEL: kernel:
; CHECK: stack_adjust 3084
; CHECK: call middle
; CHECK-NEXT: pop_exec r0l, 2
; CHECK-LABEL: middle:
; CHECK: r30
; CHECK: call leaf
; CHECK-NEXT: pop_exec r0l, 2
; CHECK: ret r1
; CHECK-LABEL: leaf:
; CHECK: stack_store {{.*}}, 8192
; CHECK: ret r1

target triple = "agx-unknown-unknown"

define agx_kernel void @kernel() {
entry:
  %x = call i32 @middle(i32 5)
  store volatile i32 %x, ptr addrspace(1) inttoptr (i64 8192 to ptr addrspace(1))
  ret void
}

define internal i32 @middle(i32 %x) {
entry:
  %a = alloca i32, align 4, addrspace(5)
  store volatile i32 %x, ptr addrspace(5) %a
  %v = load volatile i32, ptr addrspace(5) %a
  %y = call i32 @leaf(i32 %v)
  %sum = add i32 %x, %y
  ret i32 %sum
}

define internal i32 @leaf(i32 %x) {
entry:
  %a = alloca i32, align 4, addrspace(5)
  store volatile i32 %x, ptr addrspace(5) %a
  %v = load volatile i32, ptr addrspace(5) %a
  %y = add i32 %v, 3
  ret i32 %y
}
