; RUN: opt -instcombine -S -o - %s | FileCheck %s
; Check that we can replace `atomicrmw <op> LHS, 0` with `load atomic LHS`.
; This is possible when:
; - <op> LHS, 0 == LHS
; - the ordering of atomicrmw is compatible with a load (i.e., no release semantic)

; CHECK-LABEL: atomic_add_zero
; CHECK-NEXT: %res = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_add_zero(i32* %addr) {
  %res = atomicrmw add i32* %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_or_zero
; CHECK-NEXT: %res = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_or_zero(i32* %addr) {
  %res = atomicrmw add i32* %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_sub_zero
; CHECK-NEXT: %res = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_sub_zero(i32* %addr) {
  %res = atomicrmw sub i32* %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_and_allones
; CHECK-NEXT: %res = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_and_allones(i32* %addr) {
  %res = atomicrmw and i32* %addr, i32 -1 monotonic
  ret i32 %res
}
; CHECK-LABEL: atomic_umin_uint_max
; CHECK-NEXT: %res = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_umin_uint_max(i32* %addr) {
  %res = atomicrmw umin i32* %addr, i32 -1 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_umax_zero
; CHECK-NEXT: %res = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NEXT: ret i32 %res
define i32 @atomic_umax_zero(i32* %addr) {
  %res = atomicrmw umax i32* %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: atomic_min_smax_char
; CHECK-NEXT: %res = load atomic i8, i8* %addr monotonic, align 1
; CHECK-NEXT: ret i8 %res
define i8 @atomic_min_smax_char(i8* %addr) {
  %res = atomicrmw min i8* %addr, i8 127 monotonic
  ret i8 %res
}

; CHECK-LABEL: atomic_max_smin_char
; CHECK-NEXT: %res = load atomic i8, i8* %addr monotonic, align 1
; CHECK-NEXT: ret i8 %res
define i8 @atomic_max_smin_char(i8* %addr) {
  %res = atomicrmw max i8* %addr, i8 -128 monotonic
  ret i8 %res
}


; Can't replace a volatile w/a load; this would eliminate a volatile store.
; CHECK-LABEL: atomic_sub_zero_volatile
; CHECK-NEXT: %res = atomicrmw volatile sub i64* %addr, i64 0 acquire
; CHECK-NEXT: ret i64 %res
define i64 @atomic_sub_zero_volatile(i64* %addr) {
  %res = atomicrmw volatile sub i64* %addr, i64 0 acquire
  ret i64 %res
}


; Check that the transformation properly preserve the syncscope.
; CHECK-LABEL: atomic_syncscope
; CHECK-NEXT: %res = load atomic i16, i16* %addr syncscope("some_syncscope") acquire, align 2
; CHECK-NEXT: ret i16 %res
define i16 @atomic_syncscope(i16* %addr) {
  %res = atomicrmw or i16* %addr, i16 0 syncscope("some_syncscope") acquire
  ret i16 %res
}

; By eliminating the store part of the atomicrmw, we would get rid of the
; release semantic, which is incorrect.  We can canonicalize the operation.
; CHECK-LABEL: atomic_seq_cst
; CHECK-NEXT: %res = atomicrmw or i16* %addr, i16 0 seq_cst
; CHECK-NEXT: ret i16 %res
define i16 @atomic_seq_cst(i16* %addr) {
  %res = atomicrmw add i16* %addr, i16 0 seq_cst
  ret i16 %res
}

; Check that the transformation does not apply when the value is changed by
; the atomic operation (non zero constant).
; CHECK-LABEL: atomic_add_non_zero
; CHECK-NEXT: %res = atomicrmw add i16* %addr, i16 2 monotonic
; CHECK-NEXT: ret i16 %res
define i16 @atomic_add_non_zero(i16* %addr) {
  %res = atomicrmw add i16* %addr, i16 2 monotonic
  ret i16 %res
}

; CHECK-LABEL: atomic_xor_zero
; CHECK-NEXT: %res = load atomic i16, i16* %addr monotonic, align 2
; CHECK-NEXT: ret i16 %res
define i16 @atomic_xor_zero(i16* %addr) {
  %res = atomicrmw xor i16* %addr, i16 0 monotonic
  ret i16 %res
}

; Check that the transformation does not apply when the ordering is
; incompatible with a load (release).  Do canonicalize.
; CHECK-LABEL: atomic_release
; CHECK-NEXT: %res = atomicrmw or i16* %addr, i16 0 release
; CHECK-NEXT: ret i16 %res
define i16 @atomic_release(i16* %addr) {
  %res = atomicrmw sub i16* %addr, i16 0 release
  ret i16 %res
}

; Check that the transformation does not apply when the ordering is
; incompatible with a load (acquire, release).  Do canonicalize.
; CHECK-LABEL: atomic_acq_rel
; CHECK-NEXT: %res = atomicrmw or i16* %addr, i16 0 acq_rel
; CHECK-NEXT: ret i16 %res
define i16 @atomic_acq_rel(i16* %addr) {
  %res = atomicrmw xor i16* %addr, i16 0 acq_rel
  ret i16 %res
}


; CHECK-LABEL: sat_or_allones
; CHECK-NEXT: %res = atomicrmw add i32* %addr, i32 -1 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_or_allones(i32* %addr) {
  %res = atomicrmw add i32* %addr, i32 -1 monotonic
  ret i32 %res
}

; CHECK-LABEL: sat_and_zero
; CHECK-NEXT: %res = atomicrmw and i32* %addr, i32 0 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_and_zero(i32* %addr) {
  %res = atomicrmw and i32* %addr, i32 0 monotonic
  ret i32 %res
}
; CHECK-LABEL: sat_umin_uint_min
; CHECK-NEXT: %res = atomicrmw umin i32* %addr, i32 0 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_umin_uint_min(i32* %addr) {
  %res = atomicrmw umin i32* %addr, i32 0 monotonic
  ret i32 %res
}

; CHECK-LABEL: sat_umax_uint_max
; CHECK-NEXT: %res = atomicrmw umax i32* %addr, i32 -1 monotonic
; CHECK-NEXT: ret i32 %res
define i32 @sat_umax_uint_max(i32* %addr) {
  %res = atomicrmw umax i32* %addr, i32 -1 monotonic
  ret i32 %res
}

; CHECK-LABEL: sat_min_smin_char
; CHECK-NEXT: %res = atomicrmw min i8* %addr, i8 -128 monotonic
; CHECK-NEXT: ret i8 %res
define i8 @sat_min_smin_char(i8* %addr) {
  %res = atomicrmw min i8* %addr, i8 -128 monotonic
  ret i8 %res
}

; CHECK-LABEL: sat_max_smax_char
; CHECK-NEXT: %res = atomicrmw max i8* %addr, i8 127 monotonic
; CHECK-NEXT: ret i8 %res
define i8 @sat_max_smax_char(i8* %addr) {
  %res = atomicrmw max i8* %addr, i8 127 monotonic
  ret i8 %res
}

; CHECK-LABEL: xchg_unused_monotonic
; CHECK-NEXT: atomicrmw xchg i32* %addr, i32 0 monotonic
; CHECK-NEXT: ret void
define void @xchg_unused_monotonic(i32* %addr) {
  atomicrmw xchg i32* %addr, i32 0 monotonic
  ret void
}

; CHECK-LABEL: xchg_unused_release
; CHECK-NEXT: atomicrmw xchg i32* %addr, i32 -1 release
; CHECK-NEXT: ret void
define void @xchg_unused_release(i32* %addr) {
  atomicrmw xchg i32* %addr, i32 -1 release
  ret void
}

; CHECK-LABEL: xchg_unused_seq_cst
; CHECK-NEXT: atomicrmw xchg i32* %addr, i32 0 seq_cst
; CHECK-NEXT: ret void
define void @xchg_unused_seq_cst(i32* %addr) {
  atomicrmw xchg i32* %addr, i32 0 seq_cst
  ret void
}

; CHECK-LABEL: xchg_unused_volatile
; CHECK-NEXT: atomicrmw volatile xchg i32* %addr, i32 0 monotonic
; CHECK-NEXT: ret void
define void @xchg_unused_volatile(i32* %addr) {
  atomicrmw volatile xchg i32* %addr, i32 0 monotonic
  ret void
}




