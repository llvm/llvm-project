; RUN: opt -passes=instcombine -S -o - %s | FileCheck %s
; Check that we can replace `atomicrmw <op> LHS, 0` with `load atomic LHS`.
; This is possible when:
; - <op> LHS, 0 == LHS
; - the ordering of atomicrmw is compatible with a load (i.e., no release semantic)

; CHECK-LABEL: atomic_add_zero
; CHECK-NEXT: %res = load atomic i32, ptr %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_add_zero(ptr %addr) {
  %res = atomicrmw add ptr %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_or_zero
; CHECK-NEXT: %res = load atomic i32, ptr %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_or_zero(ptr %addr) {
  %res = atomicrmw add ptr %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_sub_zero
; CHECK-NEXT: %res = load atomic i32, ptr %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_sub_zero(ptr %addr) {
  %res = atomicrmw sub ptr %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_and_allones
; CHECK-NEXT: %res = load atomic i32, ptr %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_and_allones(ptr %addr) {
  %res = atomicrmw and ptr %addr, i32 -1 monotonic
  ret i32 %res
}
; CHECK-LABEL: atomic_umin_uint_max
; CHECK-NEXT: %res = load atomic i32, ptr %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_umin_uint_max(ptr %addr) {
  %res = atomicrmw umin ptr %addr, i32 -1 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_umax_zero
; CHECK-NEXT: %res = load atomic i32, ptr %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_umax_zero(ptr %addr) {
  %res = atomicrmw umax ptr %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_min_smax_char
; CHECK-NEXT: %res = load atomic i8, ptr %addr monotonic, align 1
; CHECK-NEXT: ret i8 %res
define i8 @atomic_min_smax_char(ptr %addr) {
  %res = atomicrmw min ptr %addr, i8 127 monotonic
  ret i8 %res
}

; CHECK-LABEL: atomic_max_smin_char
; CHECK-NEXT: %res = load atomic i8, ptr %addr monotonic, align 1
; CHECK-NEXT: ret i8 %res
define i8 @atomic_max_smin_char(ptr %addr) {
  %res = atomicrmw max ptr %addr, i8 -128 monotonic
  ret i8 %res
}

; CHECK-LABEL: atomic_fsub
; CHECK-NEXT: %res = load atomic float, ptr %addr monotonic, align 4
; CHECK-NEXT: ret float %res
define float @atomic_fsub_zero(ptr %addr) {
  %res = atomicrmw fsub ptr %addr, float 0.0 monotonic
  ret float %res
}

; CHECK-LABEL: atomic_fadd
; CHECK-NEXT: %res = load atomic float, ptr %addr monotonic, align 4
; CHECK-NEXT: ret float %res
define float @atomic_fadd_zero(ptr %addr) {
  %res = atomicrmw fadd ptr %addr, float -0.0 monotonic
  ret float %res
}

; CHECK-LABEL: atomic_fsub_canon
; CHECK-NEXT: %res = atomicrmw fadd ptr %addr, float -0.000000e+00 release
; CHECK-NEXT: ret float %res
define float @atomic_fsub_canon(ptr %addr) {
  %res = atomicrmw fsub ptr %addr, float 0.0 release
  ret float %res
}
; CHECK-LABEL: atomic_fadd_canon
; CHECK-NEXT: %res = atomicrmw fadd ptr %addr, float -0.000000e+00 release
; CHECK-NEXT: ret float %res
define float @atomic_fadd_canon(ptr %addr) {
  %res = atomicrmw fadd ptr %addr, float -0.0 release
  ret float %res
}

; Can't replace a volatile w/a load; this would eliminate a volatile store.
; CHECK-LABEL: atomic_sub_zero_volatile
; CHECK-NEXT: %res = atomicrmw volatile sub ptr %addr, i64 0 acquire
; CHECK-NEXT: ret i64 %res
define i64 @atomic_sub_zero_volatile(ptr %addr) {
  %res = atomicrmw volatile sub ptr %addr, i64 0 acquire
  ret i64 %res
}


; Check that the transformation properly preserve the syncscope.
; CHECK-LABEL: atomic_syncscope
; CHECK-NEXT: %res = load atomic i16, ptr %addr syncscope("some_syncscope") acquire, align 2
; CHECK-NEXT: ret i16 %res
define i16 @atomic_syncscope(ptr %addr) {
  %res = atomicrmw or ptr %addr, i16 0 syncscope("some_syncscope") acquire
  ret i16 %res
}

; By eliminating the store part of the atomicrmw, we would get rid of the
; release semantic, which is incorrect.  We can canonicalize the operation.
; CHECK-LABEL: atomic_seq_cst
; CHECK-NEXT: %res = atomicrmw or ptr %addr, i16 0 seq_cst
; CHECK-NEXT: ret i16 %res
define i16 @atomic_seq_cst(ptr %addr) {
  %res = atomicrmw add ptr %addr, i16 0 seq_cst
  ret i16 %res
}

; Check that the transformation does not apply when the value is changed by
; the atomic operation (non zero constant).
; CHECK-LABEL: atomic_add_non_zero
; CHECK-NEXT: %res = atomicrmw add ptr %addr, i16 2 monotonic
; CHECK-NEXT: ret i16 %res
define i16 @atomic_add_non_zero(ptr %addr) {
  %res = atomicrmw add ptr %addr, i16 2 monotonic
  ret i16 %res
}

; CHECK-LABEL: atomic_xor_zero
; CHECK-NEXT: %res = load atomic i16, ptr %addr monotonic, align 2
; CHECK-NEXT: ret i16 %res
define i16 @atomic_xor_zero(ptr %addr) {
  %res = atomicrmw xor ptr %addr, i16 0 monotonic
  ret i16 %res
}

; Check that the transformation does not apply when the ordering is
; incompatible with a load (release).  Do canonicalize.
; CHECK-LABEL: atomic_release
; CHECK-NEXT: %res = atomicrmw or ptr %addr, i16 0 release
; CHECK-NEXT: ret i16 %res
define i16 @atomic_release(ptr %addr) {
  %res = atomicrmw sub ptr %addr, i16 0 release
  ret i16 %res
}

; Check that the transformation does not apply when the ordering is
; incompatible with a load (acquire, release).  Do canonicalize.
; CHECK-LABEL: atomic_acq_rel
; CHECK-NEXT: %res = atomicrmw or ptr %addr, i16 0 acq_rel
; CHECK-NEXT: ret i16 %res
define i16 @atomic_acq_rel(ptr %addr) {
  %res = atomicrmw xor ptr %addr, i16 0 acq_rel
  ret i16 %res
}


; CHECK-LABEL: sat_or_allones
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, i32 -1 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_or_allones(ptr %addr) {
  %res = atomicrmw or ptr %addr, i32 -1 monotonic
  ret i32 %res
}

; CHECK-LABEL: sat_and_zero
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, i32 0 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_and_zero(ptr %addr) {
  %res = atomicrmw and ptr %addr, i32 0 monotonic
  ret i32 %res
}
; CHECK-LABEL: sat_umin_uint_min
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, i32 0 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_umin_uint_min(ptr %addr) {
  %res = atomicrmw umin ptr %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: sat_umax_uint_max
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, i32 -1 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_umax_uint_max(ptr %addr) {
  %res = atomicrmw umax ptr %addr, i32 -1 monotonic
  ret i32 %res
}

; CHECK-LABEL: sat_min_smin_char
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, i8 -128 monotonic
; CHECK-NEXT: ret i8 %res
define i8 @sat_min_smin_char(ptr %addr) {
  %res = atomicrmw min ptr %addr, i8 -128 monotonic
  ret i8 %res
}

; CHECK-LABEL: sat_max_smax_char
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, i8 127 monotonic
; CHECK-NEXT: ret i8 %res
define i8 @sat_max_smax_char(ptr %addr) {
  %res = atomicrmw max ptr %addr, i8 127 monotonic
  ret i8 %res
}

; CHECK-LABEL: sat_fadd_nan
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, double 0x7FF00000FFFFFFFF release
; CHECK-NEXT: ret double %res
define double @sat_fadd_nan(ptr %addr) {
  %res = atomicrmw fadd ptr %addr, double 0x7FF00000FFFFFFFF release
  ret double %res
}

; CHECK-LABEL: sat_fsub_nan
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, double 0x7FF00000FFFFFFFF release
; CHECK-NEXT: ret double %res
define double @sat_fsub_nan(ptr %addr) {
  %res = atomicrmw fsub ptr %addr, double 0x7FF00000FFFFFFFF release
  ret double %res
}

; CHECK-LABEL: sat_fsub_nan_unused
; CHECK-NEXT: store atomic double 0x7FF00000FFFFFFFF, ptr %addr monotonic, align 8
; CHECK-NEXT: ret void
define void @sat_fsub_nan_unused(ptr %addr) {
  atomicrmw fsub ptr %addr, double 0x7FF00000FFFFFFFF monotonic
  ret void
}

; CHECK-LABEL: xchg_unused_monotonic
; CHECK-NEXT: store atomic i32 0, ptr %addr monotonic, align 4
; CHECK-NEXT: ret void
define void @xchg_unused_monotonic(ptr %addr) {
  atomicrmw xchg ptr %addr, i32 0 monotonic
  ret void
}

; CHECK-LABEL: xchg_unused_release
; CHECK-NEXT: store atomic i32 -1, ptr %addr release, align 4
; CHECK-NEXT: ret void
define void @xchg_unused_release(ptr %addr) {
  atomicrmw xchg ptr %addr, i32 -1 release
  ret void
}

; CHECK-LABEL: xchg_unused_seq_cst
; CHECK-NEXT: atomicrmw xchg ptr %addr, i32 0 seq_cst
; CHECK-NEXT: ret void
define void @xchg_unused_seq_cst(ptr %addr) {
  atomicrmw xchg ptr %addr, i32 0 seq_cst
  ret void
}

; CHECK-LABEL: xchg_unused_volatile
; CHECK-NEXT: atomicrmw volatile xchg ptr %addr, i32 0 monotonic
; CHECK-NEXT: ret void
define void @xchg_unused_volatile(ptr %addr) {
  atomicrmw volatile xchg ptr %addr, i32 0 monotonic
  ret void
}

; CHECK-LABEL: sat_or_allones_unused
; CHECK-NEXT: store atomic i32 -1, ptr %addr monotonic, align 4
; CHECK-NEXT: ret void
define void @sat_or_allones_unused(ptr %addr) {
  atomicrmw or ptr %addr, i32 -1 monotonic
  ret void
}


; CHECK-LABEL: undef_operand_unused
; CHECK-NEXT: atomicrmw or ptr %addr, i32 undef monotonic
; CHECK-NEXT: ret void
define void @undef_operand_unused(ptr %addr) {
  atomicrmw or ptr %addr, i32 undef monotonic
  ret void
}

; CHECK-LABEL: undef_operand_used
; CHECK-NEXT: %res = atomicrmw or ptr %addr, i32 undef monotonic
; CHECK-NEXT: ret i32 %res
define i32 @undef_operand_used(ptr %addr) {
  %res = atomicrmw or ptr %addr, i32 undef monotonic
  ret i32 %res
}

; CHECK-LABEL: sat_fmax_inf
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, double 0x7FF0000000000000 monotonic
; CHECK-NEXT: ret double %res
define double @sat_fmax_inf(ptr %addr) {
  %res = atomicrmw fmax ptr %addr, double 0x7FF0000000000000 monotonic
  ret double %res
}

; CHECK-LABEL: no_sat_fmax_inf
; CHECK-NEXT: %res = atomicrmw fmax ptr %addr, double 1.000000e-01 monotonic
; CHECK-NEXT: ret double %res
define double @no_sat_fmax_inf(ptr %addr) {
  %res = atomicrmw fmax ptr %addr, double 1.000000e-01 monotonic
  ret double %res
}

; CHECK-LABEL: sat_fmin_inf
; CHECK-NEXT: %res = atomicrmw xchg ptr %addr, double 0xFFF0000000000000 monotonic
; CHECK-NEXT: ret double %res
define double @sat_fmin_inf(ptr %addr) {
  %res = atomicrmw fmin ptr %addr, double 0xFFF0000000000000 monotonic
  ret double %res
}

; CHECK-LABEL: no_sat_fmin_inf
; CHECK-NEXT: %res = atomicrmw fmin ptr %addr, double 1.000000e-01 monotonic
; CHECK-NEXT: ret double %res
define double @no_sat_fmin_inf(ptr %addr) {
  %res = atomicrmw fmin ptr %addr, double 1.000000e-01 monotonic
  ret double %res
}
