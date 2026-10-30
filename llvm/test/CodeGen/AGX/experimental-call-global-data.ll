; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
; A function kept as a call that reads global data gets the global data
; buffer's address as a parameter (AGXGlobalDataArgs): only the kernel has it,
; in a uniform register. The kernel passes it; the callee adds the variable's
; offset in the buffer.

@table = internal addrspace(1) global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4
@other = internal addrspace(1) global i32 5, align 4

; CHECK-LABEL: kernel:
; CHECK-DAG: bitop 0101, r4, u6, 0
; CHECK-DAG: bitop 0101, r5, u7, 0
; CHECK: call lookup
; CHECK: device_load 0, i32, x, {{r[0-9]+}}, u6_u7, 4, signed
define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %i) {
  %a = call i32 @lookup(i32 %i)
  %b = load i32, ptr addrspace(1) @other, align 4
  %s = add i32 %a, %b
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: lookup:
; CHECK: device_load 0, i32, x, r2, r4_r5, r2, signed
; CHECK: ret
define internal i32 @lookup(i32 %i) noinline {
  %p = getelementptr [4 x i32], ptr addrspace(1) @table, i32 0, i32 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}
