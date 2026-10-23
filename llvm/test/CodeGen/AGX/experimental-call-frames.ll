; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; The caller and leaf both spill, but use disjoint scratch regions. The kernel
; reserves both regions once and the leaf does not change stack_adjust.
; CHECK-LABEL: kernel:
; CHECK: stack_adjust 2060
; CHECK: stack_store {{.*}}, 8
; CHECK: call leaf
; CHECK-NEXT: pop_exec r0l, 2
; CHECK: stack_load {{.*}}, 8
; CHECK-LABEL: leaf:
; CHECK-NOT: stack_adjust
; CHECK: stack_store {{.*}}, 4096
; CHECK: stack_load {{.*}}, 4096
; CHECK: ret r1

target triple = "agx-unknown-unknown"
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
  %a = alloca [16 x i32], align 4, addrspace(5)
  %p0 = getelementptr [16 x i32], ptr addrspace(5) %a, i32 0, i32 0
  %p1 = getelementptr [16 x i32], ptr addrspace(5) %a, i32 0, i32 1
  %x1 = add i32 %x, 1
  store volatile i32 %x, ptr addrspace(5) %p0
  store volatile i32 %x1, ptr addrspace(5) %p1
  %v0 = load volatile i32, ptr addrspace(5) %p0
  %v1 = load volatile i32, ptr addrspace(5) %p1
  %sum = add i32 %v0, %v1
  ret i32 %sum
}
attributes #0 = { noinline }
