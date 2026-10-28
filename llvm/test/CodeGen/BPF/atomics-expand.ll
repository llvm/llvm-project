; RUN: llc -mtriple=bpfel -mcpu=v4 < %s | FileCheck %s
; RUN: not --crash llc -mtriple=bpfel -mcpu=v3 < %s 2>&1 | FileCheck --check-prefix=V3 %s
;
; Atomic operations that BPF has no insn for.
;
; The loops start with a relaxed atomic load and a seq_cst load is load acquire.
; v3 has neither.
; V3: unsupported atomic operation, please use 32/64 bit version

; 8 and 16 bits: cmpxchg of the word that they are in.
define i8 @swap8(ptr %p, i8 %v) {
; CHECK-LABEL: swap8:
; CHECK:       w0 = cmpxchg32_32(r{{[0-9]}} + 0, w0, w{{[0-9]}})
  %r = atomicrmw xchg ptr %p, i8 %v seq_cst
  ret i8 %r
}

define i16 @add16(ptr %p, i16 %v) {
; CHECK-LABEL: add16:
; CHECK:       w0 = cmpxchg32_32(r{{[0-9]}} + 0, w0, w{{[0-9]}})
  %r = atomicrmw add ptr %p, i16 %v acq_rel
  ret i16 %r
}

define i8 @cas8(ptr %p, i8 %old, i8 %new) {
; CHECK-LABEL: cas8:
; CHECK:       w0 = cmpxchg32_32(r{{[0-9]}} + 0, w0, w{{[0-9]}})
  %pair = cmpxchg ptr %p, i8 %old, i8 %new seq_cst seq_cst
  %r = extractvalue { i8, i1 } %pair, 0
  ret i8 %r
}

; No loop for the bitwise ones.
define i8 @or8(ptr %p, i8 %v) {
; CHECK-LABEL: or8:
; CHECK:       w{{[0-9]}} = atomic_fetch_or((u32 *)(r{{[0-9]}} + 0), w{{[0-9]}})
; CHECK-NOT:   cmpxchg
  %r = atomicrmw or ptr %p, i8 %v seq_cst
  ret i8 %r
}

; Operations that there is no insn for.
define i64 @max64(ptr %p, i64 %v) {
; CHECK-LABEL: max64:
; CHECK:       r0 = cmpxchg_64(r{{[0-9]}} + 0, r0, r{{[0-9]}})
  %r = atomicrmw max ptr %p, i64 %v seq_cst
  ret i64 %r
}

define i32 @nand32(ptr %p, i32 %v) {
; CHECK-LABEL: nand32:
; CHECK:       w0 = cmpxchg32_32(r{{[0-9]}} + 0, w0, w{{[0-9]}})
  %r = atomicrmw nand ptr %p, i32 %v seq_cst
  ret i32 %r
}

; seq_cst: xchg and load acquire.
define void @store64(ptr %p, i64 %v) {
; CHECK-LABEL: store64:
; CHECK:       r{{[0-9]}} = xchg_64(r1 + 0, r{{[0-9]}})
  store atomic i64 %v, ptr %p seq_cst, align 8
  ret void
}

define i64 @load64(ptr %p) {
; CHECK-LABEL: load64:
; CHECK:       r0 = load_acquire((u64 *)(r1 + 0))
  %r = load atomic i64, ptr %p seq_cst, align 8
  ret i64 %r
}

define i8 @load8(ptr %p) {
; CHECK-LABEL: load8:
; CHECK:       w0 = load_acquire((u8 *)(r1 + 0))
  %r = load atomic i8, ptr %p seq_cst, align 1
  ret i8 %r
}

; A fence: an atomic insn that returns the old value is fully ordered.
define void @fence_acquire(ptr %p) {
; CHECK-LABEL: fence_acquire:
; CHECK:       *(u64 *)(r1 + 0) = 1
; CHECK:       *(u64 *)(r10 - 8) = 0
; CHECK:       r{{[0-9]}} = atomic_fetch_add((u64 *)(r10 - 8), r{{[0-9]}})
; CHECK:       *(u64 *)(r1 + 8) = 2
  store i64 1, ptr %p
  fence acquire
  %q = getelementptr i64, ptr %p, i64 1
  store i64 2, ptr %q
  ret void
}

define void @fence_compiler() {
; CHECK-LABEL: fence_compiler:
; CHECK-NOT:   atomic
; CHECK:       exit
  fence syncscope("singlethread") seq_cst
  ret void
}
