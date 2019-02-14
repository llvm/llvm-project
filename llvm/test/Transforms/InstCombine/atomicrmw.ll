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

; Don't transform volatile atomicrmw. This would eliminate a volatile store
; otherwise.
; CHECK-LABEL: atomic_sub_zero_volatile
; CHECK-NEXT: %res = atomicrmw volatile sub i64* %addr, i64 0 acquire
; CHECK-NEXT: ret i64 %res
define i64 @atomic_sub_zero_volatile(i64* %addr) {
  %res = atomicrmw volatile sub i64* %addr, i64 0 acquire
  ret i64 %res
}


; Check that the transformation properly preserve the syncscope.
; CHECK-LABEL: atomic_or_zero
; CHECK-NEXT: %res = load atomic i16, i16* %addr syncscope("some_syncscope") acquire, align 2
; CHECK-NEXT: ret i16 %res
define i16 @atomic_or_zero(i16* %addr) {
  %res = atomicrmw or i16* %addr, i16 0 syncscope("some_syncscope") acquire
  ret i16 %res
}

; Don't transform seq_cst ordering.
; By eliminating the store part of the atomicrmw, we would get rid of the
; release semantic, which is incorrect.
; CHECK-LABEL: atomic_or_zero_seq_cst
; CHECK-NEXT: %res = atomicrmw or i16* %addr, i16 0 seq_cst
; CHECK-NEXT: ret i16 %res
define i16 @atomic_or_zero_seq_cst(i16* %addr) {
  %res = atomicrmw or i16* %addr, i16 0 seq_cst
  ret i16 %res
}

; Check that the transformation does not apply when the value is changed by
; the atomic operation (non zero constant).
; CHECK-LABEL: atomic_or_non_zero
; CHECK-NEXT: %res = atomicrmw or i16* %addr, i16 2 monotonic
; CHECK-NEXT: ret i16 %res
define i16 @atomic_or_non_zero(i16* %addr) {
  %res = atomicrmw or i16* %addr, i16 2 monotonic
  ret i16 %res
}

; Check that the transformation does not apply when the value is changed by
; the atomic operation (xor operation with zero).
; CHECK-LABEL: atomic_xor_zero
; CHECK-NEXT: %res = atomicrmw xor i16* %addr, i16 0 monotonic
; CHECK-NEXT: ret i16 %res
define i16 @atomic_xor_zero(i16* %addr) {
  %res = atomicrmw xor i16* %addr, i16 0 monotonic
  ret i16 %res
}

; Check that the transformation does not apply when the ordering is
; incompatible with a load (release).
; CHECK-LABEL: atomic_or_zero_release
; CHECK-NEXT: %res = atomicrmw or i16* %addr, i16 0 release
; CHECK-NEXT: ret i16 %res
define i16 @atomic_or_zero_release(i16* %addr) {
  %res = atomicrmw or i16* %addr, i16 0 release
  ret i16 %res
}

; Check that the transformation does not apply when the ordering is
; incompatible with a load (acquire, release).
; CHECK-LABEL: atomic_or_zero_acq_rel
; CHECK-NEXT: %res = atomicrmw or i16* %addr, i16 0 acq_rel
; CHECK-NEXT: ret i16 %res
define i16 @atomic_or_zero_acq_rel(i16* %addr) {
  %res = atomicrmw or i16* %addr, i16 0 acq_rel
  ret i16 %res
}
