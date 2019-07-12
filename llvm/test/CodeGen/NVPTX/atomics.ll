; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s


; CHECK-LABEL: atom0
define i32 @atom0(i32* %addr, i32 %val) {
; CHECK: atom.add.u32
  %ret = atomicrmw add i32* %addr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom1
define i64 @atom1(i64* %addr, i64 %val) {
; CHECK: atom.add.u64
  %ret = atomicrmw add i64* %addr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom2
define i32 @atom2(i32* %subr, i32 %val) {
; CHECK: neg.s32
; CHECK: atom.add.u32
  %ret = atomicrmw sub i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom3
define i64 @atom3(i64* %subr, i64 %val) {
; CHECK: neg.s64
; CHECK: atom.add.u64
  %ret = atomicrmw sub i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom4
define i32 @atom4(i32* %subr, i32 %val) {
; CHECK: atom.and.b32
  %ret = atomicrmw and i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom5
define i64 @atom5(i64* %subr, i64 %val) {
; CHECK: atom.and.b64
  %ret = atomicrmw and i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

;; NAND not yet supported
;define i32 @atom6(i32* %subr, i32 %val) {
;  %ret = atomicrmw nand i32* %subr, i32 %val seq_cst
;  ret i32 %ret
;}

;define i64 @atom7(i64* %subr, i64 %val) {
;  %ret = atomicrmw nand i64* %subr, i64 %val seq_cst
;  ret i64 %ret
;}

; CHECK-LABEL: atom8
define i32 @atom8(i32* %subr, i32 %val) {
; CHECK: atom.or.b32
  %ret = atomicrmw or i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom9
define i64 @atom9(i64* %subr, i64 %val) {
; CHECK: atom.or.b64
  %ret = atomicrmw or i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom10
define i32 @atom10(i32* %subr, i32 %val) {
; CHECK: atom.xor.b32
  %ret = atomicrmw xor i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom11
define i64 @atom11(i64* %subr, i64 %val) {
; CHECK: atom.xor.b64
  %ret = atomicrmw xor i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom12
define i32 @atom12(i32* %subr, i32 %val) {
; CHECK: atom.max.s32
  %ret = atomicrmw max i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom13
define i64 @atom13(i64* %subr, i64 %val) {
; CHECK: atom.max.s64
  %ret = atomicrmw max i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom14
define i32 @atom14(i32* %subr, i32 %val) {
; CHECK: atom.min.s32
  %ret = atomicrmw min i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom15
define i64 @atom15(i64* %subr, i64 %val) {
; CHECK: atom.min.s64
  %ret = atomicrmw min i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom16
define i32 @atom16(i32* %subr, i32 %val) {
; CHECK: atom.max.u32
  %ret = atomicrmw umax i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom17
define i64 @atom17(i64* %subr, i64 %val) {
; CHECK: atom.max.u64
  %ret = atomicrmw umax i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

; CHECK-LABEL: atom18
define i32 @atom18(i32* %subr, i32 %val) {
; CHECK: atom.min.u32
  %ret = atomicrmw umin i32* %subr, i32 %val seq_cst
  ret i32 %ret
}

; CHECK-LABEL: atom19
define i64 @atom19(i64* %subr, i64 %val) {
; CHECK: atom.min.u64
  %ret = atomicrmw umin i64* %subr, i64 %val seq_cst
  ret i64 %ret
}

declare float @llvm.nvvm.atomic.load.add.f32.p0f32(float* %addr, float %val)

; CHECK-LABEL: atomic_add_f32_generic
define float @atomic_add_f32_generic(float* %addr, float %val) {
; CHECK: atom.add.f32
  %ret = call float @llvm.nvvm.atomic.load.add.f32.p0f32(float* %addr, float %val)
  ret float %ret
}

declare float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* %addr, float %val)

; CHECK-LABEL: atomic_add_f32_addrspace1
define float @atomic_add_f32_addrspace1(float addrspace(1)* %addr, float %val) {
; CHECK: atom.global.add.f32
  %ret = call float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* %addr, float %val)
  ret float %ret
}

declare float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* %addr, float %val)

; CHECK-LABEL: atomic_add_f32_addrspace3
define float @atomic_add_f32_addrspace3(float addrspace(3)* %addr, float %val) {
; CHECK: atom.shared.add.f32
  %ret = call float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* %addr, float %val)
  ret float %ret
}

; CHECK-LABEL: atomicrmw_add_f32_generic
define float @atomicrmw_add_f32_generic(float* %addr, float %val) {
; CHECK: atom.add.f32
  %ret = atomicrmw fadd float* %addr, float %val seq_cst
  ret float %ret
}

; CHECK-LABEL: atomicrmw_add_f32_addrspace1
define float @atomicrmw_add_f32_addrspace1(float addrspace(1)* %addr, float %val) {
; CHECK: atom.global.add.f32
  %ret = atomicrmw fadd float addrspace(1)* %addr, float %val seq_cst
  ret float %ret
}

; CHECK-LABEL: atomicrmw_add_f32_addrspace3
define float @atomicrmw_add_f32_addrspace3(float addrspace(3)* %addr, float %val) {
; CHECK: atom.shared.add.f32
  %ret = atomicrmw fadd float addrspace(3)* %addr, float %val seq_cst
  ret float %ret
}

; CHECK-LABEL: atomic_cmpxchg_i32
define i32 @atomic_cmpxchg_i32(i32* %addr, i32 %cmp, i32 %new) {
; CHECK: atom.cas.b32
  %pairold = cmpxchg i32* %addr, i32 %cmp, i32 %new seq_cst seq_cst
  ret i32 %new
}

; CHECK-LABEL: atomic_cmpxchg_i64
define i64 @atomic_cmpxchg_i64(i64* %addr, i64 %cmp, i64 %new) {
; CHECK: atom.cas.b64
  %pairold = cmpxchg i64* %addr, i64 %cmp, i64 %new seq_cst seq_cst
  ret i64 %new
}
