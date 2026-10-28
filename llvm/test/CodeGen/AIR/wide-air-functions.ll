; The portable passes name an intrinsic's AIR function after its type, so
; i128 helpers (ZLUDA links some into every module) arrive as air.clz.i128 or
; air.abs.s.i128, which the AIR writer cannot write ("unhandled wide integer
; user 'call'"; 110 PTX-formalization programs). The Metal pipeline turns
; them back into the intrinsics and splits those into i64 limbs.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: llc -mtriple=air -filetype=obj %s -o /dev/null

; CHECK-LABEL: define void @clz(
; CHECK-NOT: i128
; CHECK: call i64 @llvm.ctlz.i64(
; CHECK-LABEL: define void @abs(
; CHECK-NOT: i128
; CHECK: icmp slt i64
define void @clz(ptr addrspace(1) %p) {
  %x = load i128, ptr addrspace(1) %p
  %r = call i128 @air.clz.i128(i128 %x, i1 false)
  store i128 %r, ptr addrspace(1) %p
  ret void
}
define void @abs(ptr addrspace(1) %p) {
  %x = load i128, ptr addrspace(1) %p
  %r = call i128 @air.abs.s.i128(i128 %x)
  store i128 %r, ptr addrspace(1) %p
  ret void
}
declare i128 @air.clz.i128(i128, i1)
declare i128 @air.abs.s.i128(i128)
