; Generic pointer emulation (generic-pointers.ll) in a module that casts no
; threadgroup or private address to generic: every generic address is a device
; address, and generic accesses are device accesses with no test.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s

; CHECK-LABEL: define void @device(
; CHECK-NOT: generic.
; CHECK: load i32, ptr addrspace(1)
; CHECK-NOT: generic.
; CHECK: ret void
define void @device(ptr addrspace(1) %o) "air-kernel" {
  %x = load i64, ptr addrspace(1) %o
  %p = inttoptr i64 %x to ptr addrspace(4)
  %v = load i32, ptr addrspace(4) %p
  store i32 %v, ptr addrspace(1) %o
  ret void
}
