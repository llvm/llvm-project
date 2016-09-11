; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s

; Check that invariant loads from the global addrspace are lowered to
; ld.global.nc.

; CHECK-LABEL: @ld_global
define i32 @ld_global(i32 addrspace(1)* %ptr) {
; CHECK: ld.global.nc.{{[a-z]}}32
  %a = load i32, i32 addrspace(1)* %ptr, !invariant.load !0
  ret i32 %a
}

; CHECK-LABEL: @ld_not_invariant
define i32 @ld_not_invariant(i32 addrspace(1)* %ptr) {
; CHECK: ld.global.{{[a-z]}}32
  %a = load i32, i32 addrspace(1)* %ptr
  ret i32 %a
}

; CHECK-LABEL: @ld_not_global_addrspace
define i32 @ld_not_global_addrspace(i32 addrspace(0)* %ptr) {
; CHECK: ld.{{[a-z]}}32
  %a = load i32, i32 addrspace(0)* %ptr
  ret i32 %a
}

!0 = !{}
