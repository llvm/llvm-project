; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; An i64 and three-element array are split across r2..r6; the three-element
; result returns in r2..r4. The mask pop immediately follows the call.
; CHECK-LABEL: kernel:
; CHECK: call leaf
; CHECK-NEXT: pop_exec r0l, 2
; CHECK-LABEL: leaf:
; CHECK: iadd r2
; CHECK: ret r1

target triple = "agx-unknown-unknown"
define agx_kernel void @kernel() {
entry:
  %res = call [3 x i32] @leaf(i64 4294967299, [3 x i32] [i32 10, i32 20, i32 30], i32 5)
  %a = extractvalue [3 x i32] %res, 0
  %b = extractvalue [3 x i32] %res, 1
  %c = extractvalue [3 x i32] %res, 2
  store volatile i32 %a, ptr addrspace(1) inttoptr (i64 8192 to ptr addrspace(1))
  store volatile i32 %b, ptr addrspace(1) inttoptr (i64 8196 to ptr addrspace(1))
  store volatile i32 %c, ptr addrspace(1) inttoptr (i64 8200 to ptr addrspace(1))
  ret void
}
define internal [3 x i32] @leaf(i64 %a, [3 x i32] %v, i32 %bias) #0 {
entry:
  %low = trunc i64 %a to i32
  %v0 = extractvalue [3 x i32] %v, 0
  %v1 = extractvalue [3 x i32] %v, 1
  %v2 = extractvalue [3 x i32] %v, 2
  %r0 = add i32 %v0, %bias
  %r1 = add i32 %v1, %low
  %r2 = add i32 %v2, 1
  %o0 = insertvalue [3 x i32] poison, i32 %r0, 0
  %o1 = insertvalue [3 x i32] %o0, i32 %r1, 1
  %o2 = insertvalue [3 x i32] %o1, i32 %r2, 2
  ret [3 x i32] %o2
}
attributes #0 = { noinline }
