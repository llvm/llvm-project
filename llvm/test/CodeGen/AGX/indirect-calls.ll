; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-always-inline < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; An indirect call tests the callee against each address-taken function of
; its type and calls it directly (the calls are inlined later); none
; matching traps. Function addresses as values are the functions' numbers.

; CHECK: @table = internal addrspace(1) global [3 x ptr] [ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 2 to ptr), ptr inttoptr (i64 3 to ptr)]
@table = internal addrspace(1) global [3 x ptr] [ptr @twice, ptr @square, ptr @other]

define internal i32 @twice(i32 %x) {
  %r = shl i32 %x, 1
  ret i32 %r
}

define internal i32 @square(i32 %x) {
  %r = mul i32 %x, %x
  ret i32 %r
}

define internal float @other(float %x) {
  ret float %x
}

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: icmp eq ptr %f, inttoptr (i64 1 to ptr)
; CHECK: call i32 @twice(
; CHECK: icmp eq ptr %f, inttoptr (i64 2 to ptr)
; CHECK: call i32 @square(
; CHECK-NOT: call i32 %f
; CHECK-NOT: @other(
; CHECK: call void @llvm.trap()
; ASM-LABEL: k:
; ASM-NOT: error
define agx_kernel void @k(ptr addrspace(1) %o, i32 %i, i32 %x) {
  %pf = getelementptr [3 x ptr], ptr addrspace(1) @table, i32 0, i32 %i
  %f = load ptr, ptr addrspace(1) %pf
  %r = call i32 %f(i32 %x)
  store i32 %r, ptr addrspace(1) %o
  ret void
}
