; Generic pointer emulation (generic-pointers.ll) in a module that casts no
; threadgroup address to generic: no generic address is one, so an access of
; unknown origin tests only the private tag.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s

; CHECK-LABEL: define void @unknown(
; CHECK-NOT: generic.shared
; CHECK: lshr i64 {{%.*}}, 62
; CHECK: generic.device:
; CHECK: generic.private:
define void @unknown(ptr addrspace(1) %o) "air-kernel" {
  %a = alloca i32
  %ga = addrspacecast ptr %a to ptr addrspace(4)
  %ia = ptrtoint ptr addrspace(4) %ga to i64
  store i64 %ia, ptr addrspace(1) %o
  %x = load i64, ptr addrspace(1) %o
  %p = inttoptr i64 %x to ptr addrspace(4)
  %v = load i32, ptr addrspace(4) %p
  store i32 %v, ptr addrspace(1) %o
  ret void
}

