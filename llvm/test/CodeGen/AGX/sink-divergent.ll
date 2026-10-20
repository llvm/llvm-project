; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A value computed before a divergent if / else and used after it must not
; be sunk into the Flow block between the two sides: that block runs with
; only the lanes of the first side active, so the others would read a stale
; register.

declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: k:
; CHECK: bfeil [[V:r[0-9]+]], 0, {{r[0-9]+}}, 7
; CHECK: if_icmp
; CHECK: else_
; CHECK: device_store 0, i32, x, [[V]]
define agx_kernel void @k(ptr addrspace(1) %out, ptr addrspace(1) %a) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr i32, ptr addrspace(1) %a, i32 %t
  %x = load i32, ptr addrspace(1) %pa
  %v = lshr i32 %x, 7
  %c = icmp ult i32 %t, 16
  br i1 %c, label %then, label %else
then:
  store i32 1, ptr addrspace(1) %pa
  br label %join
else:
  store i32 2, ptr addrspace(1) %pa
  br label %join
join:
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %po
  ret void
}
