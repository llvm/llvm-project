; RUN: llc < %s -march=nvptx -mcpu=sm_32 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_32 | %ptxas-verify %}


; CHECK-LABEL: atom0
define i32 @atom0(ptr %addr, i32 %val) {
; CHECK: atom.add.u32
  %ret = atomicrmw add ptr %addr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom1
define i64 @atom1(ptr %addr, i64 %val) {
; CHECK: atom.add.u64
  %ret = atomicrmw add ptr %addr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom2
define i32 @atom2(ptr %subr, i32 %val) {
; CHECK: neg.s32
; CHECK: atom.add.u32
  %ret = atomicrmw sub ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom3
define i64 @atom3(ptr %subr, i64 %val) {
; CHECK: neg.s64
; CHECK: atom.add.u64
  %ret = atomicrmw sub ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom4
define i32 @atom4(ptr %subr, i32 %val) {
; CHECK: atom.and.b32
  %ret = atomicrmw and ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom5
define i64 @atom5(ptr %subr, i64 %val) {
; CHECK: atom.and.b64
  %ret = atomicrmw and ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

;; NAND not yet supported
;define i32 @atom6(ptr %subr, i32 %val) {
;  %ret = atomicrmw nand ptr %subr, i32 %val seq_cst
;  ret i32 %ret
;}

;define i64 @atom7(ptr %subr, i64 %val) {
;  %ret = atomicrmw nand ptr %subr, i64 %val seq_cst
;  ret i64 %ret
;}

; CHECK-LABEL: atom8
define i32 @atom8(ptr %subr, i32 %val) {
; CHECK: atom.or.b32
  %ret = atomicrmw or ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom9
define i64 @atom9(ptr %subr, i64 %val) {
; CHECK: atom.or.b64
  %ret = atomicrmw or ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom10
define i32 @atom10(ptr %subr, i32 %val) {
; CHECK: atom.xor.b32
  %ret = atomicrmw xor ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom11
define i64 @atom11(ptr %subr, i64 %val) {
; CHECK: atom.xor.b64
  %ret = atomicrmw xor ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom12
define i32 @atom12(ptr %subr, i32 %val) {
; CHECK: atom.max.s32
  %ret = atomicrmw max ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom13
define i64 @atom13(ptr %subr, i64 %val) {
; CHECK: atom.max.s64
  %ret = atomicrmw max ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom14
define i32 @atom14(ptr %subr, i32 %val) {
; CHECK: atom.min.s32
  %ret = atomicrmw min ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom15
define i64 @atom15(ptr %subr, i64 %val) {
; CHECK: atom.min.s64
  %ret = atomicrmw min ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom16
define i32 @atom16(ptr %subr, i32 %val) {
; CHECK: atom.max.u32
  %ret = atomicrmw umax ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom17
define i64 @atom17(ptr %subr, i64 %val) {
; CHECK: atom.max.u64
  %ret = atomicrmw umax ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom18
define i32 @atom18(ptr %subr, i32 %val) {
; CHECK: atom.min.u32
  %ret = atomicrmw umin ptr %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom19
define i64 @atom19(ptr %subr, i64 %val) {
; CHECK: atom.min.u64
  %ret = atomicrmw umin ptr %subr, i64 %val seq_cst
  ret i64 %ret
}

declare float @llvm.nvvm.atomic.load.add.f32.p0(ptr %addr, float %val)

; CHECK-LABEL: atomic_add_f32_generic
define float @atomic_add_f32_generic(ptr %addr, float %val) {
; CHECK: atom.add.f32
  %ret = call float @llvm.nvvm.atomic.load.add.f32.p0(ptr %addr, float %val)
  ret float %ret
}

declare float @llvm.nvvm.atomic.load.add.f32.p1(ptr addrspace(1) %addr, float %val)

; CHECK-LABEL: atomic_add_f32_addrspace1
define float @atomic_add_f32_addrspace1(ptr addrspace(1) %addr, float %val) {
; CHECK: atom.global.add.f32
  %ret = call float @llvm.nvvm.atomic.load.add.f32.p1(ptr addrspace(1) %addr, float %val)
  ret float %ret
}

declare float @llvm.nvvm.atomic.load.add.f32.p3(ptr addrspace(3) %addr, float %val)

; CHECK-LABEL: atomic_add_f32_addrspace3
define float @atomic_add_f32_addrspace3(ptr addrspace(3) %addr, float %val) {
; CHECK: atom.shared.add.f32
  %ret = call float @llvm.nvvm.atomic.load.add.f32.p3(ptr addrspace(3) %addr, float %val)
  ret float %ret
}

; CHECK-LABEL: atomicrmw_add_f32_generic
define float @atomicrmw_add_f32_generic(ptr %addr, float %val) {
; CHECK: atom.add.f32
  %ret = atomicrmw fadd ptr %addr, float %val seq_cst
  ret float %ret
}

; CHECK-LABEL: atomicrmw_add_f32_addrspace1
define float @atomicrmw_add_f32_addrspace1(ptr addrspace(1) %addr, float %val) {
; CHECK: atom.global.add.f32
  %ret = atomicrmw fadd ptr addrspace(1) %addr, float %val seq_cst
  ret float %ret
}

; CHECK-LABEL: atomicrmw_add_f32_addrspace3
define float @atomicrmw_add_f32_addrspace3(ptr addrspace(3) %addr, float %val) {
; CHECK: atom.shared.add.f32
  %ret = atomicrmw fadd ptr addrspace(3) %addr, float %val seq_cst
  ret float %ret
}

; CHECK-LABEL: atomic_cmpxchg_i32
define i32 @atomic_cmpxchg_i32(ptr %addr, i32 %cmp, i32 %new) {
; CHECK: atom.cas.b32
  %pairold = cmpxchg ptr %addr, i32 %cmp, i32 %new seq_cst seq_cst
  ret i32 %new
}

; CHECK-LABEL: atomic_cmpxchg_i64
define i64 @atomic_cmpxchg_i64(ptr %addr, i64 %cmp, i64 %new) {
; CHECK: atom.cas.b64
  %pairold = cmpxchg ptr %addr, i64 %cmp, i64 %new seq_cst seq_cst
  ret i64 %new
}
