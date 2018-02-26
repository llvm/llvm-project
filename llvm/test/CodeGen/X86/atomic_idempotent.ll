; RUN: llc < %s -mtriple=x86_64-- -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=X64
; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=X32

; On x86, an atomic rmw operation that does not modify the value in memory
; (such as atomic add 0) can be replaced by an mfence followed by a mov.
; This is explained (with the motivation for such an optimization) in
; http://www.hpl.hp.com/techreports/2012/HPL-2012-68.pdf

define i8 @add8(i8* %p) {
; CHECK-LABEL: add8
; CHECK: mfence
; CHECK: movb
  %1 = atomicrmw add i8* %p, i8 0 monotonic
  ret i8 %1
}

define i16 @or16(i16* %p) {
; CHECK-LABEL: or16
; CHECK: mfence
; CHECK: movw
  %1 = atomicrmw or i16* %p, i16 0 acquire
  ret i16 %1
}

define i32 @xor32(i32* %p) {
; CHECK-LABEL: xor32
; CHECK: mfence
; CHECK: movl
  %1 = atomicrmw xor i32* %p, i32 0 release
  ret i32 %1
}

define i64 @sub64(i64* %p) {
; CHECK-LABEL: sub64
; X64: mfence
; X64: movq
; X32-NOT: mfence
  %1 = atomicrmw sub i64* %p, i64 0 seq_cst
  ret i64 %1
}

define i128 @or128(i128* %p) {
; CHECK-LABEL: or128
; CHECK-NOT: mfence
  %1 = atomicrmw or i128* %p, i128 0 monotonic
  ret i128 %1
}

; For 'and', the idempotent value is (-1)
define i32 @and32 (i32* %p) {
; CHECK-LABEL: and32
; CHECK: mfence
; CHECK: movl
  %1 = atomicrmw and i32* %p, i32 -1 acq_rel
  ret i32 %1
}
