; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s

; Check that invariant loads from the global addrspace are lowered to
; ld.global.nc.

; CHECK-LABEL: @ld_global
define i32 @ld_global(i32 addrspace(1)* %ptr) {
; CHECK: ld.global.nc.{{[a-z]}}32
  %a = load i32, i32 addrspace(1)* %ptr, !invariant.load !0
  ret i32 %a
}

; CHECK-LABEL: @ld_global_v2i32
define i32 @ld_global_v2i32(<2 x i32> addrspace(1)* %ptr) {
; CHECK: ld.global.nc.v2.{{[a-z]}}32
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %ptr, !invariant.load !0
  %v1 = extractelement <2 x i32> %a, i32 0
  %v2 = extractelement <2 x i32> %a, i32 1
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

; CHECK-LABEL: @ld_global_v4i32
define i32 @ld_global_v4i32(<4 x i32> addrspace(1)* %ptr) {
; CHECK: ld.global.nc.v4.{{[a-z]}}32
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %ptr, !invariant.load !0
  %v1 = extractelement <4 x i32> %a, i32 0
  %v2 = extractelement <4 x i32> %a, i32 1
  %v3 = extractelement <4 x i32> %a, i32 2
  %v4 = extractelement <4 x i32> %a, i32 3
  %sum1 = add i32 %v1, %v2
  %sum2 = add i32 %v3, %v4
  %sum3 = add i32 %sum1, %sum2
  ret i32 %sum3
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
