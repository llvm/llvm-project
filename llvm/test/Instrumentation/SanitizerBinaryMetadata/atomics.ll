; RUN: opt < %s -passes='module(sanmd-module)' -sanitizer-metadata-atomics -S | FileCheck %s

; Check that atomic memory operations receive PC sections metadata.

; CHECK: @__start_sanmd_atomics = extern_weak hidden global ptr
; CHECK: @__stop_sanmd_atomics = extern_weak hidden global ptr
; CHECK: @__start_sanmd_covered = extern_weak hidden global ptr
; CHECK: @__stop_sanmd_covered = extern_weak hidden global ptr

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i8 @atomic8_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a unordered, align 1
  ret i8 %0
}
; CHECK-LABEL: atomic8_load_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i8 @atomic8_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a monotonic, align 1
  ret i8 %0
}
; CHECK-LABEL: atomic8_load_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i8 @atomic8_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a acquire, align 1
  ret i8 %0
}
; CHECK-LABEL: atomic8_load_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i8 @atomic8_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a seq_cst, align 1
  ret i8 %0
}
; CHECK-LABEL: atomic8_load_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a unordered, align 1
  ret void
}
; CHECK-LABEL: atomic8_store_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a monotonic, align 1
  ret void
}
; CHECK-LABEL: atomic8_store_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a release, align 1
  ret void
}
; CHECK-LABEL: atomic8_store_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a seq_cst, align 1
  ret void
}
; CHECK-LABEL: atomic8_store_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 monotonic
  ret void
}
; CHECK-LABEL: atomic8_xchg_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 monotonic
  ret void
}
; CHECK-LABEL: atomic8_add_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 monotonic
  ret void
}
; CHECK-LABEL: atomic8_sub_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 monotonic
  ret void
}
; CHECK-LABEL: atomic8_and_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 monotonic
  ret void
}
; CHECK-LABEL: atomic8_or_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 monotonic
  ret void
}
; CHECK-LABEL: atomic8_xor_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 monotonic
  ret void
}
; CHECK-LABEL: atomic8_nand_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 acquire
  ret void
}
; CHECK-LABEL: atomic8_xchg_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 acquire
  ret void
}
; CHECK-LABEL: atomic8_add_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 acquire
  ret void
}
; CHECK-LABEL: atomic8_sub_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 acquire
  ret void
}
; CHECK-LABEL: atomic8_and_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 acquire
  ret void
}
; CHECK-LABEL: atomic8_or_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 acquire
  ret void
}
; CHECK-LABEL: atomic8_xor_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 acquire
  ret void
}
; CHECK-LABEL: atomic8_nand_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 release
  ret void
}
; CHECK-LABEL: atomic8_xchg_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 release
  ret void
}
; CHECK-LABEL: atomic8_add_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 release
  ret void
}
; CHECK-LABEL: atomic8_sub_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 release
  ret void
}
; CHECK-LABEL: atomic8_and_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 release
  ret void
}
; CHECK-LABEL: atomic8_or_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 release
  ret void
}
; CHECK-LABEL: atomic8_xor_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 release
  ret void
}
; CHECK-LABEL: atomic8_nand_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic8_xchg_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic8_add_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic8_sub_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic8_and_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic8_or_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic8_xor_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic8_nand_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic8_xchg_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic8_add_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic8_sub_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic8_and_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic8_or_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic8_xor_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic8_nand_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic8_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 monotonic monotonic
  cmpxchg ptr %a, i8 0, i8 1 monotonic acquire
  cmpxchg ptr %a, i8 0, i8 1 monotonic seq_cst
  ret void
}
; CHECK-LABEL: atomic8_cas_monotonic{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i8 0, i8 1 monotonic monotonic, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 monotonic acquire, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 monotonic seq_cst, align 1, !pcsections !2

define void @atomic8_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 acquire monotonic
  cmpxchg ptr %a, i8 0, i8 1 acquire acquire
  cmpxchg ptr %a, i8 0, i8 1 acquire seq_cst
  ret void
}
; CHECK-LABEL: atomic8_cas_acquire{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i8 0, i8 1 acquire monotonic, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 acquire acquire, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 acquire seq_cst, align 1, !pcsections !2

define void @atomic8_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 release monotonic
  cmpxchg ptr %a, i8 0, i8 1 release acquire
  cmpxchg ptr %a, i8 0, i8 1 release seq_cst
  ret void
}
; CHECK-LABEL: atomic8_cas_release{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i8 0, i8 1 release monotonic, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 release acquire, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 release seq_cst, align 1, !pcsections !2

define void @atomic8_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 acq_rel monotonic
  cmpxchg ptr %a, i8 0, i8 1 acq_rel acquire
  cmpxchg ptr %a, i8 0, i8 1 acq_rel seq_cst
  ret void
}
; CHECK-LABEL: atomic8_cas_acq_rel{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i8 0, i8 1 acq_rel monotonic, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 acq_rel acquire, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 acq_rel seq_cst, align 1, !pcsections !2

define void @atomic8_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 seq_cst monotonic
  cmpxchg ptr %a, i8 0, i8 1 seq_cst acquire
  cmpxchg ptr %a, i8 0, i8 1 seq_cst seq_cst
  ret void
}
; CHECK-LABEL: atomic8_cas_seq_cst{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i8 0, i8 1 seq_cst monotonic, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 seq_cst acquire, align 1, !pcsections !2
; CHECK: cmpxchg ptr %a, i8 0, i8 1 seq_cst seq_cst, align 1, !pcsections !2

define i16 @atomic16_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a unordered, align 2
  ret i16 %0
}
; CHECK-LABEL: atomic16_load_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i16 @atomic16_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a monotonic, align 2
  ret i16 %0
}
; CHECK-LABEL: atomic16_load_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i16 @atomic16_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a acquire, align 2
  ret i16 %0
}
; CHECK-LABEL: atomic16_load_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i16 @atomic16_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a seq_cst, align 2
  ret i16 %0
}
; CHECK-LABEL: atomic16_load_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a unordered, align 2
  ret void
}
; CHECK-LABEL: atomic16_store_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a monotonic, align 2
  ret void
}
; CHECK-LABEL: atomic16_store_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a release, align 2
  ret void
}
; CHECK-LABEL: atomic16_store_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a seq_cst, align 2
  ret void
}
; CHECK-LABEL: atomic16_store_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 monotonic
  ret void
}
; CHECK-LABEL: atomic16_xchg_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 monotonic
  ret void
}
; CHECK-LABEL: atomic16_add_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 monotonic
  ret void
}
; CHECK-LABEL: atomic16_sub_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 monotonic
  ret void
}
; CHECK-LABEL: atomic16_and_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 monotonic
  ret void
}
; CHECK-LABEL: atomic16_or_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 monotonic
  ret void
}
; CHECK-LABEL: atomic16_xor_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 monotonic
  ret void
}
; CHECK-LABEL: atomic16_nand_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 acquire
  ret void
}
; CHECK-LABEL: atomic16_xchg_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 acquire
  ret void
}
; CHECK-LABEL: atomic16_add_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 acquire
  ret void
}
; CHECK-LABEL: atomic16_sub_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 acquire
  ret void
}
; CHECK-LABEL: atomic16_and_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 acquire
  ret void
}
; CHECK-LABEL: atomic16_or_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 acquire
  ret void
}
; CHECK-LABEL: atomic16_xor_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 acquire
  ret void
}
; CHECK-LABEL: atomic16_nand_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 release
  ret void
}
; CHECK-LABEL: atomic16_xchg_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 release
  ret void
}
; CHECK-LABEL: atomic16_add_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 release
  ret void
}
; CHECK-LABEL: atomic16_sub_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 release
  ret void
}
; CHECK-LABEL: atomic16_and_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 release
  ret void
}
; CHECK-LABEL: atomic16_or_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 release
  ret void
}
; CHECK-LABEL: atomic16_xor_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 release
  ret void
}
; CHECK-LABEL: atomic16_nand_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic16_xchg_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic16_add_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic16_sub_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic16_and_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic16_or_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic16_xor_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic16_nand_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic16_xchg_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic16_add_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic16_sub_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic16_and_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic16_or_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic16_xor_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic16_nand_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic16_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 monotonic monotonic
  cmpxchg ptr %a, i16 0, i16 1 monotonic acquire
  cmpxchg ptr %a, i16 0, i16 1 monotonic seq_cst
  ret void
}
; CHECK-LABEL: atomic16_cas_monotonic{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i16 0, i16 1 monotonic monotonic, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 monotonic acquire, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 monotonic seq_cst, align 2, !pcsections !2

define void @atomic16_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 acquire monotonic
  cmpxchg ptr %a, i16 0, i16 1 acquire acquire
  cmpxchg ptr %a, i16 0, i16 1 acquire seq_cst
  ret void
}
; CHECK-LABEL: atomic16_cas_acquire{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i16 0, i16 1 acquire monotonic, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 acquire acquire, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 acquire seq_cst, align 2, !pcsections !2

define void @atomic16_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 release monotonic
  cmpxchg ptr %a, i16 0, i16 1 release acquire
  cmpxchg ptr %a, i16 0, i16 1 release seq_cst
  ret void
}
; CHECK-LABEL: atomic16_cas_release{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i16 0, i16 1 release monotonic, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 release acquire, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 release seq_cst, align 2, !pcsections !2

define void @atomic16_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 acq_rel monotonic
  cmpxchg ptr %a, i16 0, i16 1 acq_rel acquire
  cmpxchg ptr %a, i16 0, i16 1 acq_rel seq_cst
  ret void
}
; CHECK-LABEL: atomic16_cas_acq_rel{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i16 0, i16 1 acq_rel monotonic, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 acq_rel acquire, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 acq_rel seq_cst, align 2, !pcsections !2

define void @atomic16_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 seq_cst monotonic
  cmpxchg ptr %a, i16 0, i16 1 seq_cst acquire
  cmpxchg ptr %a, i16 0, i16 1 seq_cst seq_cst
  ret void
}
; CHECK-LABEL: atomic16_cas_seq_cst{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i16 0, i16 1 seq_cst monotonic, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 seq_cst acquire, align 2, !pcsections !2
; CHECK: cmpxchg ptr %a, i16 0, i16 1 seq_cst seq_cst, align 2, !pcsections !2

define i32 @atomic32_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a unordered, align 4
  ret i32 %0
}
; CHECK-LABEL: atomic32_load_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i32 @atomic32_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a monotonic, align 4
  ret i32 %0
}
; CHECK-LABEL: atomic32_load_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i32 @atomic32_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a acquire, align 4
  ret i32 %0
}
; CHECK-LABEL: atomic32_load_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i32 @atomic32_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a seq_cst, align 4
  ret i32 %0
}
; CHECK-LABEL: atomic32_load_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a unordered, align 4
  ret void
}
; CHECK-LABEL: atomic32_store_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a monotonic, align 4
  ret void
}
; CHECK-LABEL: atomic32_store_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a release, align 4
  ret void
}
; CHECK-LABEL: atomic32_store_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a seq_cst, align 4
  ret void
}
; CHECK-LABEL: atomic32_store_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 monotonic
  ret void
}
; CHECK-LABEL: atomic32_xchg_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 monotonic
  ret void
}
; CHECK-LABEL: atomic32_add_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 monotonic
  ret void
}
; CHECK-LABEL: atomic32_sub_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 monotonic
  ret void
}
; CHECK-LABEL: atomic32_and_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 monotonic
  ret void
}
; CHECK-LABEL: atomic32_or_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 monotonic
  ret void
}
; CHECK-LABEL: atomic32_xor_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 monotonic
  ret void
}
; CHECK-LABEL: atomic32_nand_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 acquire
  ret void
}
; CHECK-LABEL: atomic32_xchg_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 acquire
  ret void
}
; CHECK-LABEL: atomic32_add_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 acquire
  ret void
}
; CHECK-LABEL: atomic32_sub_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 acquire
  ret void
}
; CHECK-LABEL: atomic32_and_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 acquire
  ret void
}
; CHECK-LABEL: atomic32_or_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 acquire
  ret void
}
; CHECK-LABEL: atomic32_xor_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 acquire
  ret void
}
; CHECK-LABEL: atomic32_nand_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 release
  ret void
}
; CHECK-LABEL: atomic32_xchg_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 release
  ret void
}
; CHECK-LABEL: atomic32_add_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 release
  ret void
}
; CHECK-LABEL: atomic32_sub_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 release
  ret void
}
; CHECK-LABEL: atomic32_and_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 release
  ret void
}
; CHECK-LABEL: atomic32_or_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 release
  ret void
}
; CHECK-LABEL: atomic32_xor_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 release
  ret void
}
; CHECK-LABEL: atomic32_nand_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic32_xchg_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic32_add_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic32_sub_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic32_and_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic32_or_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic32_xor_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic32_nand_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic32_xchg_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic32_add_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic32_sub_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic32_and_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic32_or_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic32_xor_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic32_nand_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic32_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 monotonic monotonic
  cmpxchg ptr %a, i32 0, i32 1 monotonic acquire
  cmpxchg ptr %a, i32 0, i32 1 monotonic seq_cst
  ret void
}
; CHECK-LABEL: atomic32_cas_monotonic{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i32 0, i32 1 monotonic monotonic, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 monotonic acquire, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 monotonic seq_cst, align 4, !pcsections !2

define void @atomic32_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 acquire monotonic
  cmpxchg ptr %a, i32 0, i32 1 acquire acquire
  cmpxchg ptr %a, i32 0, i32 1 acquire seq_cst
  ret void
}
; CHECK-LABEL: atomic32_cas_acquire{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i32 0, i32 1 acquire monotonic, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 acquire acquire, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 acquire seq_cst, align 4, !pcsections !2

define void @atomic32_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 release monotonic
  cmpxchg ptr %a, i32 0, i32 1 release acquire
  cmpxchg ptr %a, i32 0, i32 1 release seq_cst
  ret void
}
; CHECK-LABEL: atomic32_cas_release{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i32 0, i32 1 release monotonic, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 release acquire, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 release seq_cst, align 4, !pcsections !2

define void @atomic32_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 acq_rel monotonic
  cmpxchg ptr %a, i32 0, i32 1 acq_rel acquire
  cmpxchg ptr %a, i32 0, i32 1 acq_rel seq_cst
  ret void
}
; CHECK-LABEL: atomic32_cas_acq_rel{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i32 0, i32 1 acq_rel monotonic, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 acq_rel acquire, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 acq_rel seq_cst, align 4, !pcsections !2

define void @atomic32_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 seq_cst monotonic
  cmpxchg ptr %a, i32 0, i32 1 seq_cst acquire
  cmpxchg ptr %a, i32 0, i32 1 seq_cst seq_cst
  ret void
}
; CHECK-LABEL: atomic32_cas_seq_cst{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i32 0, i32 1 seq_cst monotonic, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 seq_cst acquire, align 4, !pcsections !2
; CHECK: cmpxchg ptr %a, i32 0, i32 1 seq_cst seq_cst, align 4, !pcsections !2

define i64 @atomic64_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a unordered, align 8
  ret i64 %0
}
; CHECK-LABEL: atomic64_load_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i64 @atomic64_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a monotonic, align 8
  ret i64 %0
}
; CHECK-LABEL: atomic64_load_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i64 @atomic64_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a acquire, align 8
  ret i64 %0
}
; CHECK-LABEL: atomic64_load_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i64 @atomic64_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a seq_cst, align 8
  ret i64 %0
}
; CHECK-LABEL: atomic64_load_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define ptr @atomic64_load_seq_cst_ptr_ty(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic ptr, ptr %a seq_cst, align 8
  ret ptr %0
}
; CHECK-LABEL: atomic64_load_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a unordered, align 8
  ret void
}
; CHECK-LABEL: atomic64_store_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a monotonic, align 8
  ret void
}
; CHECK-LABEL: atomic64_store_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a release, align 8
  ret void
}
; CHECK-LABEL: atomic64_store_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a seq_cst, align 8
  ret void
}
; CHECK-LABEL: atomic64_store_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_store_seq_cst_ptr_ty(ptr %a, ptr %v) nounwind uwtable {
entry:
  store atomic ptr %v, ptr %a seq_cst, align 8
  ret void
}
; CHECK-LABEL: atomic64_store_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 monotonic
  ret void
}
; CHECK-LABEL: atomic64_xchg_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 monotonic
  ret void
}
; CHECK-LABEL: atomic64_add_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 monotonic
  ret void
}
; CHECK-LABEL: atomic64_sub_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 monotonic
  ret void
}
; CHECK-LABEL: atomic64_and_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 monotonic
  ret void
}
; CHECK-LABEL: atomic64_or_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 monotonic
  ret void
}
; CHECK-LABEL: atomic64_xor_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 monotonic
  ret void
}
; CHECK-LABEL: atomic64_nand_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 acquire
  ret void
}
; CHECK-LABEL: atomic64_xchg_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 acquire
  ret void
}
; CHECK-LABEL: atomic64_add_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 acquire
  ret void
}
; CHECK-LABEL: atomic64_sub_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 acquire
  ret void
}
; CHECK-LABEL: atomic64_and_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 acquire
  ret void
}
; CHECK-LABEL: atomic64_or_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 acquire
  ret void
}
; CHECK-LABEL: atomic64_xor_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 acquire
  ret void
}
; CHECK-LABEL: atomic64_nand_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 release
  ret void
}
; CHECK-LABEL: atomic64_xchg_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 release
  ret void
}
; CHECK-LABEL: atomic64_add_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 release
  ret void
}
; CHECK-LABEL: atomic64_sub_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 release
  ret void
}
; CHECK-LABEL: atomic64_and_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 release
  ret void
}
; CHECK-LABEL: atomic64_or_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 release
  ret void
}
; CHECK-LABEL: atomic64_xor_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 release
  ret void
}
; CHECK-LABEL: atomic64_nand_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic64_xchg_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic64_add_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic64_sub_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic64_and_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic64_or_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic64_xor_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic64_nand_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic64_xchg_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic64_add_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic64_sub_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic64_and_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic64_or_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic64_xor_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic64_nand_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic64_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 monotonic monotonic
  cmpxchg ptr %a, i64 0, i64 1 monotonic acquire
  cmpxchg ptr %a, i64 0, i64 1 monotonic seq_cst
  ret void
}
; CHECK-LABEL: atomic64_cas_monotonic{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i64 0, i64 1 monotonic monotonic, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 monotonic acquire, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 monotonic seq_cst, align 8, !pcsections !2

define void @atomic64_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 acquire monotonic
  cmpxchg ptr %a, i64 0, i64 1 acquire acquire
  cmpxchg ptr %a, i64 0, i64 1 acquire seq_cst
  ret void
}
; CHECK-LABEL: atomic64_cas_acquire{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i64 0, i64 1 acquire monotonic, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 acquire acquire, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 acquire seq_cst, align 8, !pcsections !2

define void @atomic64_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 release monotonic
  cmpxchg ptr %a, i64 0, i64 1 release acquire
  cmpxchg ptr %a, i64 0, i64 1 release seq_cst
  ret void
}
; CHECK-LABEL: atomic64_cas_release{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i64 0, i64 1 release monotonic, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 release acquire, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 release seq_cst, align 8, !pcsections !2

define void @atomic64_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 acq_rel monotonic
  cmpxchg ptr %a, i64 0, i64 1 acq_rel acquire
  cmpxchg ptr %a, i64 0, i64 1 acq_rel seq_cst
  ret void
}
; CHECK-LABEL: atomic64_cas_acq_rel{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i64 0, i64 1 acq_rel monotonic, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 acq_rel acquire, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 acq_rel seq_cst, align 8, !pcsections !2

define void @atomic64_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 seq_cst monotonic
  cmpxchg ptr %a, i64 0, i64 1 seq_cst acquire
  cmpxchg ptr %a, i64 0, i64 1 seq_cst seq_cst
  ret void
}
; CHECK-LABEL: atomic64_cas_seq_cst{{.*}}!pcsections !0
; CHECK: cmpxchg ptr %a, i64 0, i64 1 seq_cst monotonic, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 seq_cst acquire, align 8, !pcsections !2
; CHECK: cmpxchg ptr %a, i64 0, i64 1 seq_cst seq_cst, align 8, !pcsections !2

define void @atomic64_cas_seq_cst_ptr_ty(ptr %a, ptr %v1, ptr %v2) nounwind uwtable {
entry:
  cmpxchg ptr %a, ptr %v1, ptr %v2 seq_cst seq_cst
  ret void
}
; CHECK-LABEL: atomic64_cas_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i128 @atomic128_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a unordered, align 16
  ret i128 %0
}
; CHECK-LABEL: atomic128_load_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i128 @atomic128_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a monotonic, align 16
  ret i128 %0
}
; CHECK-LABEL: atomic128_load_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i128 @atomic128_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a acquire, align 16
  ret i128 %0
}
; CHECK-LABEL: atomic128_load_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define i128 @atomic128_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a seq_cst, align 16
  ret i128 %0
}
; CHECK-LABEL: atomic128_load_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a unordered, align 16
  ret void
}
; CHECK-LABEL: atomic128_store_unordered{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a monotonic, align 16
  ret void
}
; CHECK-LABEL: atomic128_store_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a release, align 16
  ret void
}
; CHECK-LABEL: atomic128_store_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a seq_cst, align 16
  ret void
}
; CHECK-LABEL: atomic128_store_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 monotonic
  ret void
}
; CHECK-LABEL: atomic128_xchg_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 monotonic
  ret void
}
; CHECK-LABEL: atomic128_add_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 monotonic
  ret void
}
; CHECK-LABEL: atomic128_sub_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 monotonic
  ret void
}
; CHECK-LABEL: atomic128_and_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 monotonic
  ret void
}
; CHECK-LABEL: atomic128_or_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 monotonic
  ret void
}
; CHECK-LABEL: atomic128_xor_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 monotonic
  ret void
}
; CHECK-LABEL: atomic128_nand_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 acquire
  ret void
}
; CHECK-LABEL: atomic128_xchg_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 acquire
  ret void
}
; CHECK-LABEL: atomic128_add_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 acquire
  ret void
}
; CHECK-LABEL: atomic128_sub_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 acquire
  ret void
}
; CHECK-LABEL: atomic128_and_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 acquire
  ret void
}
; CHECK-LABEL: atomic128_or_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 acquire
  ret void
}
; CHECK-LABEL: atomic128_xor_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 acquire
  ret void
}
; CHECK-LABEL: atomic128_nand_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 release
  ret void
}
; CHECK-LABEL: atomic128_xchg_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 release
  ret void
}
; CHECK-LABEL: atomic128_add_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 release
  ret void
}
; CHECK-LABEL: atomic128_sub_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 release
  ret void
}
; CHECK-LABEL: atomic128_and_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 release
  ret void
}
; CHECK-LABEL: atomic128_or_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 release
  ret void
}
; CHECK-LABEL: atomic128_xor_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 release
  ret void
}
; CHECK-LABEL: atomic128_nand_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic128_xchg_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic128_add_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic128_sub_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic128_and_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic128_or_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic128_xor_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 acq_rel
  ret void
}
; CHECK-LABEL: atomic128_nand_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic128_xchg_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic128_add_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic128_sub_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic128_and_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic128_or_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic128_xor_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 seq_cst
  ret void
}
; CHECK-LABEL: atomic128_nand_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 monotonic monotonic
  ret void
}
; CHECK-LABEL: atomic128_cas_monotonic{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 acquire acquire
  ret void
}
; CHECK-LABEL: atomic128_cas_acquire{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 release monotonic
  ret void
}
; CHECK-LABEL: atomic128_cas_release{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 acq_rel acquire
  ret void
}
; CHECK-LABEL: atomic128_cas_acq_rel{{.*}}!pcsections !0
; CHECK: !pcsections !2

define void @atomic128_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 seq_cst seq_cst
  ret void
}
; CHECK-LABEL: atomic128_cas_seq_cst{{.*}}!pcsections !0
; CHECK: !pcsections !2

; Check that callbacks are emitted.

; CHECK-LABEL: __sanitizer_metadata_atomics.module_ctor
; CHECK: call void @__sanitizer_metadata_atomics_add(i32 1, ptr @__start_sanmd_atomics, ptr @__stop_sanmd_atomics)

; CHECK-LABEL: __sanitizer_metadata_atomics.module_dtor
; CHECK: call void @__sanitizer_metadata_atomics_del(i32 1, ptr @__start_sanmd_atomics, ptr @__stop_sanmd_atomics)

; CHECK-LABEL: __sanitizer_metadata_covered.module_ctor
; CHECK: call void @__sanitizer_metadata_covered_add(i32 1, ptr @__start_sanmd_covered, ptr @__stop_sanmd_covered)

; CHECK-LABEL: __sanitizer_metadata_covered.module_dtor
; CHECK: call void @__sanitizer_metadata_covered_del(i32 1, ptr @__start_sanmd_covered, ptr @__stop_sanmd_covered)

; CHECK: !0 = !{!"sanmd_covered", !1}
; CHECK: !1 = !{i32 1}
; CHECK: !2 = !{!"sanmd_atomics"}
