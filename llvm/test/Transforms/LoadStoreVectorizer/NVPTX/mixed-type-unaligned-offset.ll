; RUN: opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx88 \
; RUN:     -passes=load-store-vectorizer -S < %s | FileCheck %s

; The two <2 x half> accesses are 7 bytes apart.  They only stay in one chain
; because the wider `double` access overlaps them, which is possible now that
; the equivalence-class key no longer partitions by element size.  A chain like
; this cannot be laid out on a `half`-sized (16-bit) lane grid: it spans 11
; bytes, and 8 * 11 is not a multiple of 16.  getChainElemTy must therefore
; fold the element offsets into the lane-size GCD and pick an 8-bit lane.
; This used to assert in splitChainByAlignment with
; "(8 * SizeBytes) % VecElemBits == 0".

define double @mixed_type_unaligned_offset(ptr addrspace(5) align 32 %p) {
; CHECK-LABEL: define double @mixed_type_unaligned_offset(
; CHECK-NOT: assert
; CHECK: ret double
entry:
  %g4 = getelementptr i8, ptr addrspace(5) %p, i64 9
  %l0 = load <2 x half>, ptr addrspace(5) %g4, align 1
  %g8 = getelementptr i8, ptr addrspace(5) %p, i64 16
  %l2 = load <2 x half>, ptr addrspace(5) %g8, align 16
  %g14 = getelementptr i8, ptr addrspace(5) %p, i64 8
  %l3 = load double, ptr addrspace(5) %g14, align 8
  ret double %l3
}
