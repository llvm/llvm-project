; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

; Check that invariant loads from the global addrspace are lowered to
; ld.global.nc.

; CHECK-LABEL: @ld_global
define i32 @ld_global(ptr addrspace(1) %ptr) {
; CHECK: ld.global.nc.{{[a-z]}}32
  %a = load i32, ptr addrspace(1) %ptr, !invariant.load !0
  ret i32 %a
}

; CHECK-LABEL: @ld_global_v2f16
define half @ld_global_v2f16(ptr addrspace(1) %ptr) {
; Load of v2f16 is weird. We consider it to be a legal type, which happens to be
; loaded/stored as a 32-bit scalar.
; CHECK: ld.global.nc.u32
  %a = load <2 x half>, ptr addrspace(1) %ptr, !invariant.load !0
  %v1 = extractelement <2 x half> %a, i32 0
  %v2 = extractelement <2 x half> %a, i32 1
  %sum = fadd half %v1, %v2
  ret half %sum
}

; CHECK-LABEL: @ld_global_v4f16
define half @ld_global_v4f16(ptr addrspace(1) %ptr) {
; Larger f16 vectors may be split into individual f16 elements and multiple
; loads/stores may be vectorized using f16 element type. Practically it's
; limited to v4 variant only.
; CHECK: ld.global.nc.v4.u16
  %a = load <4 x half>, ptr addrspace(1) %ptr, !invariant.load !0
  %v1 = extractelement <4 x half> %a, i32 0
  %v2 = extractelement <4 x half> %a, i32 1
  %v3 = extractelement <4 x half> %a, i32 2
  %v4 = extractelement <4 x half> %a, i32 3
  %sum1 = fadd half %v1, %v2
  %sum2 = fadd half %v3, %v4
  %sum = fadd half %sum1, %sum2
  ret half %sum
}

; CHECK-LABEL: @ld_global_v8f16
define half @ld_global_v8f16(ptr addrspace(1) %ptr) {
; Larger vectors are, again, loaded as v4i32. PTX has no v8 variants of loads/stores,
; so load/store vectorizer has to convert v8f16 -> v4 x v2f16.
; CHECK: ld.global.nc.v4.u32
  %a = load <8 x half>, ptr addrspace(1) %ptr, !invariant.load !0
  %v1 = extractelement <8 x half> %a, i32 0
  %v2 = extractelement <8 x half> %a, i32 2
  %v3 = extractelement <8 x half> %a, i32 4
  %v4 = extractelement <8 x half> %a, i32 6
  %sum1 = fadd half %v1, %v2
  %sum2 = fadd half %v3, %v4
  %sum = fadd half %sum1, %sum2
  ret half %sum
}

; CHECK-LABEL: @ld_global_v2i32
define i32 @ld_global_v2i32(ptr addrspace(1) %ptr) {
; CHECK: ld.global.nc.v2.{{[a-z]}}32
  %a = load <2 x i32>, ptr addrspace(1) %ptr, !invariant.load !0
  %v1 = extractelement <2 x i32> %a, i32 0
  %v2 = extractelement <2 x i32> %a, i32 1
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

; CHECK-LABEL: @ld_global_v4i32
define i32 @ld_global_v4i32(ptr addrspace(1) %ptr) {
; CHECK: ld.global.nc.v4.{{[a-z]}}32
  %a = load <4 x i32>, ptr addrspace(1) %ptr, !invariant.load !0
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
define i32 @ld_not_invariant(ptr addrspace(1) %ptr) {
; CHECK: ld.global.{{[a-z]}}32
  %a = load i32, ptr addrspace(1) %ptr
  ret i32 %a
}

; CHECK-LABEL: @ld_not_global_addrspace
define i32 @ld_not_global_addrspace(ptr addrspace(0) %ptr) {
; CHECK: ld.{{[a-z]}}32
  %a = load i32, ptr addrspace(0) %ptr
  ret i32 %a
}

!0 = !{}
