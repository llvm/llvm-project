; RUN: opt < %s -passes=tsan -S | FileCheck %s
; RUN: opt < %s -passes=tsan -S -mtriple=s390x-unknown-linux | FileCheck --check-prefix=EXT %s
; RUN: opt < %s -passes=tsan -S -mtriple=mips-linux-gnu | FileCheck --check-prefix=MIPS_EXT %s
; RUN: opt < %s -passes=tsan -S -mtriple=loongarch64-unknown-linux-gnu | FileCheck --check-prefix=LA_EXT %s
; REQUIRES: x86-registered-target, systemz-registered-target, mips-registered-target, loongarch-registered-target
; Check that atomic memory operations are converted to calls into ThreadSanitizer runtime.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i8 @atomic8_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a unordered, align 1, !dbg !7
  ret i8 %0, !dbg !7
}
; CHECK-LABEL: atomic8_load_unordered
; CHECK: call i8 @__tsan_atomic8_load(ptr %a, i32 0), !dbg

define i8 @atomic8_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a monotonic, align 1, !dbg !7
  ret i8 %0, !dbg !7
}
; CHECK-LABEL: atomic8_load_monotonic
; CHECK: call i8 @__tsan_atomic8_load(ptr %a, i32 0), !dbg

define i8 @atomic8_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a acquire, align 1, !dbg !7
  ret i8 %0, !dbg !7
}
; CHECK-LABEL: atomic8_load_acquire
; CHECK: call i8 @__tsan_atomic8_load(ptr %a, i32 2), !dbg

define i8 @atomic8_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a seq_cst, align 1, !dbg !7
  ret i8 %0, !dbg !7
}
; CHECK-LABEL: atomic8_load_seq_cst
; CHECK: call i8 @__tsan_atomic8_load(ptr %a, i32 5), !dbg

define void @atomic8_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a unordered, align 1, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_store_unordered
; CHECK: call void @__tsan_atomic8_store(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a monotonic, align 1, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_store_monotonic
; CHECK: call void @__tsan_atomic8_store(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a release, align 1, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_store_release
; CHECK: call void @__tsan_atomic8_store(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i8 0, ptr %a seq_cst, align 1, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_store_seq_cst
; CHECK: call void @__tsan_atomic8_store(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xchg_monotonic
; CHECK: call i8 @__tsan_atomic8_exchange(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_add_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_add(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_sub_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_sub(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_and_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_and(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_or_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_or(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xor_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_xor(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_nand_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_nand(ptr %a, i8 0, i32 0), !dbg

define void @atomic8_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xchg_acquire
; CHECK: call i8 @__tsan_atomic8_exchange(ptr %a, i8 0, i32 2), !dbg

define void @atomic8_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_add_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_add(ptr %a, i8 0, i32 2), !dbg

define void @atomic8_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_sub_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_sub(ptr %a, i8 0, i32 2), !dbg

define void @atomic8_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_and_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_and(ptr %a, i8 0, i32 2), !dbg

define void @atomic8_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_or_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_or(ptr %a, i8 0, i32 2), !dbg

define void @atomic8_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xor_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_xor(ptr %a, i8 0, i32 2), !dbg

define void @atomic8_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_nand_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_nand(ptr %a, i8 0, i32 2), !dbg

define void @atomic8_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xchg_release
; CHECK: call i8 @__tsan_atomic8_exchange(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_add_release
; CHECK: call i8 @__tsan_atomic8_fetch_add(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_sub_release
; CHECK: call i8 @__tsan_atomic8_fetch_sub(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_and_release
; CHECK: call i8 @__tsan_atomic8_fetch_and(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_or_release
; CHECK: call i8 @__tsan_atomic8_fetch_or(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xor_release
; CHECK: call i8 @__tsan_atomic8_fetch_xor(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_nand_release
; CHECK: call i8 @__tsan_atomic8_fetch_nand(ptr %a, i8 0, i32 3), !dbg

define void @atomic8_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xchg_acq_rel
; CHECK: call i8 @__tsan_atomic8_exchange(ptr %a, i8 0, i32 4), !dbg

define void @atomic8_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_add_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_add(ptr %a, i8 0, i32 4), !dbg

define void @atomic8_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_sub_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_sub(ptr %a, i8 0, i32 4), !dbg

define void @atomic8_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_and_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_and(ptr %a, i8 0, i32 4), !dbg

define void @atomic8_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_or_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_or(ptr %a, i8 0, i32 4), !dbg

define void @atomic8_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xor_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_xor(ptr %a, i8 0, i32 4), !dbg

define void @atomic8_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_nand_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_nand(ptr %a, i8 0, i32 4), !dbg

define void @atomic8_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i8 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xchg_seq_cst
; CHECK: call i8 @__tsan_atomic8_exchange(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i8 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_add_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_add(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i8 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_sub_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_sub(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i8 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_and_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_and(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i8 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_or_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_or(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i8 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_xor_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_xor(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i8 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_nand_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_nand(ptr %a, i8 0, i32 5), !dbg

define void @atomic8_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 monotonic monotonic, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 monotonic acquire, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 monotonic seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_cas_monotonic
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 0, i32 0), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 0, i32 2), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 0, i32 5), !dbg

define void @atomic8_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 acquire monotonic, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 acquire acquire, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 acquire seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_cas_acquire
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 2, i32 0), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 2, i32 2), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 2, i32 5), !dbg

define void @atomic8_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 release monotonic, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 release acquire, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 release seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_cas_release
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 3, i32 0), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 3, i32 2), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 3, i32 5), !dbg

define void @atomic8_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 acq_rel monotonic, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 acq_rel acquire, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 acq_rel seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_cas_acq_rel
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 4, i32 0), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 4, i32 2), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 4, i32 5), !dbg

define void @atomic8_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i8 0, i8 1 seq_cst monotonic, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 seq_cst acquire, !dbg !7
  cmpxchg ptr %a, i8 0, i8 1 seq_cst seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic8_cas_seq_cst
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 5, i32 0), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 5, i32 2), !dbg
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(ptr %a, i8 0, i8 1, i32 5, i32 5), !dbg

define i16 @atomic16_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a unordered, align 2, !dbg !7
  ret i16 %0, !dbg !7
}
; CHECK-LABEL: atomic16_load_unordered
; CHECK: call i16 @__tsan_atomic16_load(ptr %a, i32 0), !dbg

define i16 @atomic16_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a monotonic, align 2, !dbg !7
  ret i16 %0, !dbg !7
}
; CHECK-LABEL: atomic16_load_monotonic
; CHECK: call i16 @__tsan_atomic16_load(ptr %a, i32 0), !dbg

define i16 @atomic16_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a acquire, align 2, !dbg !7
  ret i16 %0, !dbg !7
}
; CHECK-LABEL: atomic16_load_acquire
; CHECK: call i16 @__tsan_atomic16_load(ptr %a, i32 2), !dbg

define i16 @atomic16_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i16, ptr %a seq_cst, align 2, !dbg !7
  ret i16 %0, !dbg !7
}
; CHECK-LABEL: atomic16_load_seq_cst
; CHECK: call i16 @__tsan_atomic16_load(ptr %a, i32 5), !dbg

define void @atomic16_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a unordered, align 2, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_store_unordered
; CHECK: call void @__tsan_atomic16_store(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a monotonic, align 2, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_store_monotonic
; CHECK: call void @__tsan_atomic16_store(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a release, align 2, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_store_release
; CHECK: call void @__tsan_atomic16_store(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i16 0, ptr %a seq_cst, align 2, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_store_seq_cst
; CHECK: call void @__tsan_atomic16_store(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xchg_monotonic
; CHECK: call i16 @__tsan_atomic16_exchange(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_add_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_add(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_sub_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_sub(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_and_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_and(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_or_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_or(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xor_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_xor(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_nand_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_nand(ptr %a, i16 0, i32 0), !dbg

define void @atomic16_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xchg_acquire
; CHECK: call i16 @__tsan_atomic16_exchange(ptr %a, i16 0, i32 2), !dbg

define void @atomic16_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_add_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_add(ptr %a, i16 0, i32 2), !dbg

define void @atomic16_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_sub_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_sub(ptr %a, i16 0, i32 2), !dbg

define void @atomic16_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_and_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_and(ptr %a, i16 0, i32 2), !dbg

define void @atomic16_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_or_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_or(ptr %a, i16 0, i32 2), !dbg

define void @atomic16_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xor_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_xor(ptr %a, i16 0, i32 2), !dbg

define void @atomic16_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_nand_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_nand(ptr %a, i16 0, i32 2), !dbg

define void @atomic16_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xchg_release
; CHECK: call i16 @__tsan_atomic16_exchange(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_add_release
; CHECK: call i16 @__tsan_atomic16_fetch_add(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_sub_release
; CHECK: call i16 @__tsan_atomic16_fetch_sub(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_and_release
; CHECK: call i16 @__tsan_atomic16_fetch_and(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_or_release
; CHECK: call i16 @__tsan_atomic16_fetch_or(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xor_release
; CHECK: call i16 @__tsan_atomic16_fetch_xor(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_nand_release
; CHECK: call i16 @__tsan_atomic16_fetch_nand(ptr %a, i16 0, i32 3), !dbg

define void @atomic16_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xchg_acq_rel
; CHECK: call i16 @__tsan_atomic16_exchange(ptr %a, i16 0, i32 4), !dbg

define void @atomic16_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_add_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_add(ptr %a, i16 0, i32 4), !dbg

define void @atomic16_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_sub_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_sub(ptr %a, i16 0, i32 4), !dbg

define void @atomic16_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_and_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_and(ptr %a, i16 0, i32 4), !dbg

define void @atomic16_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_or_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_or(ptr %a, i16 0, i32 4), !dbg

define void @atomic16_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xor_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_xor(ptr %a, i16 0, i32 4), !dbg

define void @atomic16_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_nand_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_nand(ptr %a, i16 0, i32 4), !dbg

define void @atomic16_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i16 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xchg_seq_cst
; CHECK: call i16 @__tsan_atomic16_exchange(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i16 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_add_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_add(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i16 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_sub_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_sub(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i16 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_and_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_and(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i16 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_or_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_or(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i16 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_xor_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_xor(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i16 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_nand_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_nand(ptr %a, i16 0, i32 5), !dbg

define void @atomic16_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 monotonic monotonic, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 monotonic acquire, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 monotonic seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_cas_monotonic
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 0, i32 0), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 0, i32 2), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 0, i32 5), !dbg

define void @atomic16_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 acquire monotonic, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 acquire acquire, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 acquire seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_cas_acquire
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 2, i32 0), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 2, i32 2), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 2, i32 5), !dbg

define void @atomic16_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 release monotonic, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 release acquire, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 release seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_cas_release
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 3, i32 0), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 3, i32 2), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 3, i32 5), !dbg

define void @atomic16_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 acq_rel monotonic, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 acq_rel acquire, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 acq_rel seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_cas_acq_rel
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 4, i32 0), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 4, i32 2), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 4, i32 5), !dbg

define void @atomic16_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i16 0, i16 1 seq_cst monotonic, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 seq_cst acquire, !dbg !7
  cmpxchg ptr %a, i16 0, i16 1 seq_cst seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic16_cas_seq_cst
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 5, i32 0), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 5, i32 2), !dbg
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(ptr %a, i16 0, i16 1, i32 5, i32 5), !dbg

define i32 @atomic32_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a unordered, align 4, !dbg !7
  ret i32 %0, !dbg !7
}
; CHECK-LABEL: atomic32_load_unordered
; CHECK: call i32 @__tsan_atomic32_load(ptr %a, i32 0), !dbg

define i32 @atomic32_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a monotonic, align 4, !dbg !7
  ret i32 %0, !dbg !7
}
; CHECK-LABEL: atomic32_load_monotonic
; CHECK: call i32 @__tsan_atomic32_load(ptr %a, i32 0), !dbg

define i32 @atomic32_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a acquire, align 4, !dbg !7
  ret i32 %0, !dbg !7
}
; CHECK-LABEL: atomic32_load_acquire
; CHECK: call i32 @__tsan_atomic32_load(ptr %a, i32 2), !dbg

define i32 @atomic32_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i32, ptr %a seq_cst, align 4, !dbg !7
  ret i32 %0, !dbg !7
}
; CHECK-LABEL: atomic32_load_seq_cst
; CHECK: call i32 @__tsan_atomic32_load(ptr %a, i32 5), !dbg

define void @atomic32_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a unordered, align 4, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_store_unordered
; CHECK: call void @__tsan_atomic32_store(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a monotonic, align 4, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_store_monotonic
; CHECK: call void @__tsan_atomic32_store(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a release, align 4, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_store_release
; CHECK: call void @__tsan_atomic32_store(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i32 0, ptr %a seq_cst, align 4, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_store_seq_cst
; CHECK: call void @__tsan_atomic32_store(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xchg_monotonic
; CHECK: call i32 @__tsan_atomic32_exchange(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_add_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_add(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_sub_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_sub(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_and_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_and(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_or_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_or(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xor_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_xor(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_nand_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_nand(ptr %a, i32 0, i32 0), !dbg

define void @atomic32_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xchg_acquire
; CHECK: call i32 @__tsan_atomic32_exchange(ptr %a, i32 0, i32 2), !dbg

define void @atomic32_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_add_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_add(ptr %a, i32 0, i32 2), !dbg

define void @atomic32_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_sub_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_sub(ptr %a, i32 0, i32 2), !dbg

define void @atomic32_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_and_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_and(ptr %a, i32 0, i32 2), !dbg

define void @atomic32_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_or_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_or(ptr %a, i32 0, i32 2), !dbg

define void @atomic32_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xor_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_xor(ptr %a, i32 0, i32 2), !dbg

define void @atomic32_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_nand_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_nand(ptr %a, i32 0, i32 2), !dbg

define void @atomic32_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xchg_release
; CHECK: call i32 @__tsan_atomic32_exchange(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_add_release
; CHECK: call i32 @__tsan_atomic32_fetch_add(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_sub_release
; CHECK: call i32 @__tsan_atomic32_fetch_sub(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_and_release
; CHECK: call i32 @__tsan_atomic32_fetch_and(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_or_release
; CHECK: call i32 @__tsan_atomic32_fetch_or(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xor_release
; CHECK: call i32 @__tsan_atomic32_fetch_xor(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_nand_release
; CHECK: call i32 @__tsan_atomic32_fetch_nand(ptr %a, i32 0, i32 3), !dbg

define void @atomic32_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xchg_acq_rel
; CHECK: call i32 @__tsan_atomic32_exchange(ptr %a, i32 0, i32 4), !dbg

define void @atomic32_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_add_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_add(ptr %a, i32 0, i32 4), !dbg

define void @atomic32_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_sub_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_sub(ptr %a, i32 0, i32 4), !dbg

define void @atomic32_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_and_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_and(ptr %a, i32 0, i32 4), !dbg

define void @atomic32_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_or_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_or(ptr %a, i32 0, i32 4), !dbg

define void @atomic32_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xor_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_xor(ptr %a, i32 0, i32 4), !dbg

define void @atomic32_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_nand_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_nand(ptr %a, i32 0, i32 4), !dbg

define void @atomic32_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i32 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xchg_seq_cst
; CHECK: call i32 @__tsan_atomic32_exchange(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i32 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_add_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_add(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i32 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_sub_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_sub(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i32 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_and_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_and(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i32 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_or_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_or(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i32 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_xor_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_xor(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i32 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_nand_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_nand(ptr %a, i32 0, i32 5), !dbg

define void @atomic32_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 monotonic monotonic, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 monotonic acquire, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 monotonic seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_cas_monotonic
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 0, i32 0), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 0, i32 2), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 0, i32 5), !dbg

define void @atomic32_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 acquire monotonic, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 acquire acquire, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 acquire seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_cas_acquire
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 2, i32 0), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 2, i32 2), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 2, i32 5), !dbg

define void @atomic32_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 release monotonic, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 release acquire, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 release seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_cas_release
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 3, i32 0), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 3, i32 2), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 3, i32 5), !dbg

define void @atomic32_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 acq_rel monotonic, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 acq_rel acquire, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 acq_rel seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_cas_acq_rel
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 4, i32 0), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 4, i32 2), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 4, i32 5), !dbg

define void @atomic32_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i32 0, i32 1 seq_cst monotonic, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 seq_cst acquire, !dbg !7
  cmpxchg ptr %a, i32 0, i32 1 seq_cst seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic32_cas_seq_cst
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 5, i32 0), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 5, i32 2), !dbg
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(ptr %a, i32 0, i32 1, i32 5, i32 5), !dbg

define i64 @atomic64_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a unordered, align 8, !dbg !7
  ret i64 %0, !dbg !7
}
; CHECK-LABEL: atomic64_load_unordered
; CHECK: call i64 @__tsan_atomic64_load(ptr %a, i32 0), !dbg

define i64 @atomic64_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a monotonic, align 8, !dbg !7
  ret i64 %0, !dbg !7
}
; CHECK-LABEL: atomic64_load_monotonic
; CHECK: call i64 @__tsan_atomic64_load(ptr %a, i32 0), !dbg

define i64 @atomic64_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a acquire, align 8, !dbg !7
  ret i64 %0, !dbg !7
}
; CHECK-LABEL: atomic64_load_acquire
; CHECK: call i64 @__tsan_atomic64_load(ptr %a, i32 2), !dbg

define i64 @atomic64_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i64, ptr %a seq_cst, align 8, !dbg !7
  ret i64 %0, !dbg !7
}
; CHECK-LABEL: atomic64_load_seq_cst
; CHECK: call i64 @__tsan_atomic64_load(ptr %a, i32 5), !dbg

define ptr @atomic64_load_seq_cst_ptr_ty(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic ptr, ptr %a seq_cst, align 8, !dbg !7
  ret ptr %0, !dbg !7
}
; CHECK-LABEL: atomic64_load_seq_cst
; CHECK: call i64 @__tsan_atomic64_load(ptr %a, i32 5), !dbg
; CHECK-NEXT: inttoptr i64 %{{.+}} to ptr

define void @atomic64_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a unordered, align 8, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_store_unordered
; CHECK: call void @__tsan_atomic64_store(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a monotonic, align 8, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_store_monotonic
; CHECK: call void @__tsan_atomic64_store(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a release, align 8, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_store_release
; CHECK: call void @__tsan_atomic64_store(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i64 0, ptr %a seq_cst, align 8, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_store_seq_cst
; CHECK: call void @__tsan_atomic64_store(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_store_seq_cst_ptr_ty(ptr %a, ptr %v) nounwind uwtable {
entry:
  store atomic ptr %v, ptr %a seq_cst, align 8, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_store_seq_cst
; CHECK: call void @__tsan_atomic64_store(ptr %a, i64 %{{.*}}, i32 5), !dbg
define void @atomic64_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xchg_monotonic
; CHECK: call i64 @__tsan_atomic64_exchange(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_add_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_add(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_sub_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_sub(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_and_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_and(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_or_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_or(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xor_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_xor(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_nand_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_nand(ptr %a, i64 0, i32 0), !dbg

define void @atomic64_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xchg_acquire
; CHECK: call i64 @__tsan_atomic64_exchange(ptr %a, i64 0, i32 2), !dbg

define void @atomic64_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_add_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_add(ptr %a, i64 0, i32 2), !dbg

define void @atomic64_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_sub_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_sub(ptr %a, i64 0, i32 2), !dbg

define void @atomic64_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_and_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_and(ptr %a, i64 0, i32 2), !dbg

define void @atomic64_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_or_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_or(ptr %a, i64 0, i32 2), !dbg

define void @atomic64_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xor_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_xor(ptr %a, i64 0, i32 2), !dbg

define void @atomic64_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_nand_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_nand(ptr %a, i64 0, i32 2), !dbg

define void @atomic64_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xchg_release
; CHECK: call i64 @__tsan_atomic64_exchange(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_add_release
; CHECK: call i64 @__tsan_atomic64_fetch_add(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_sub_release
; CHECK: call i64 @__tsan_atomic64_fetch_sub(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_and_release
; CHECK: call i64 @__tsan_atomic64_fetch_and(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_or_release
; CHECK: call i64 @__tsan_atomic64_fetch_or(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xor_release
; CHECK: call i64 @__tsan_atomic64_fetch_xor(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_nand_release
; CHECK: call i64 @__tsan_atomic64_fetch_nand(ptr %a, i64 0, i32 3), !dbg

define void @atomic64_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xchg_acq_rel
; CHECK: call i64 @__tsan_atomic64_exchange(ptr %a, i64 0, i32 4), !dbg

define void @atomic64_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_add_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_add(ptr %a, i64 0, i32 4), !dbg

define void @atomic64_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_sub_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_sub(ptr %a, i64 0, i32 4), !dbg

define void @atomic64_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_and_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_and(ptr %a, i64 0, i32 4), !dbg

define void @atomic64_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_or_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_or(ptr %a, i64 0, i32 4), !dbg

define void @atomic64_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xor_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_xor(ptr %a, i64 0, i32 4), !dbg

define void @atomic64_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_nand_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_nand(ptr %a, i64 0, i32 4), !dbg

define void @atomic64_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i64 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xchg_seq_cst
; CHECK: call i64 @__tsan_atomic64_exchange(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i64 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_add_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_add(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i64 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_sub_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_sub(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i64 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_and_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_and(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i64 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_or_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_or(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i64 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_xor_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_xor(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i64 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_nand_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_nand(ptr %a, i64 0, i32 5), !dbg

define void @atomic64_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 monotonic monotonic, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 monotonic acquire, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 monotonic seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_cas_monotonic
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 0, i32 0), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 0, i32 2), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 0, i32 5), !dbg

define void @atomic64_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 acquire monotonic, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 acquire acquire, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 acquire seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_cas_acquire
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 2, i32 0), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 2, i32 2), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 2, i32 5), !dbg

define void @atomic64_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 release monotonic, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 release acquire, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 release seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_cas_release
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 3, i32 0), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 3, i32 2), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 3, i32 5), !dbg

define void @atomic64_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 acq_rel monotonic, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 acq_rel acquire, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 acq_rel seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_cas_acq_rel
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 4, i32 0), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 4, i32 2), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 4, i32 5), !dbg

define void @atomic64_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i64 0, i64 1 seq_cst monotonic, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 seq_cst acquire, !dbg !7
  cmpxchg ptr %a, i64 0, i64 1 seq_cst seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic64_cas_seq_cst
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 5, i32 0), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 5, i32 2), !dbg
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(ptr %a, i64 0, i64 1, i32 5, i32 5), !dbg

define void @atomic64_cas_seq_cst_ptr_ty(ptr %a, ptr %v1, ptr %v2) nounwind uwtable {
entry:
  cmpxchg ptr %a, ptr %v1, ptr %v2 seq_cst seq_cst, !dbg !7
  ret void
}
; CHECK-LABEL: atomic64_cas_seq_cst
; CHECK: {{.*}} = ptrtoint ptr %v1 to i64
; CHECK-NEXT: {{.*}} = ptrtoint ptr %v2 to i64
; CHECK-NEXT: {{.*}} = call i64 @__tsan_atomic64_compare_exchange_val(ptr {{.*}}, i64 {{.*}}, i64 {{.*}}, i32 5, i32 5), !dbg
; CHECK-NEXT: {{.*}} = icmp eq i64
; CHECK-NEXT: {{.*}} = inttoptr i64 {{.*}} to ptr
; CHECK-NEXT: {{.*}} = insertvalue { ptr, i1 } poison, ptr {{.*}}, 0
; CHECK-NEXT: {{.*}} = insertvalue { ptr, i1 } {{.*}}, i1 {{.*}}, 1

define i128 @atomic128_load_unordered(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a unordered, align 16, !dbg !7
  ret i128 %0, !dbg !7
}
; CHECK-LABEL: atomic128_load_unordered
; CHECK: call i128 @__tsan_atomic128_load(ptr %a, i32 0), !dbg

define i128 @atomic128_load_monotonic(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a monotonic, align 16, !dbg !7
  ret i128 %0, !dbg !7
}
; CHECK-LABEL: atomic128_load_monotonic
; CHECK: call i128 @__tsan_atomic128_load(ptr %a, i32 0), !dbg

define i128 @atomic128_load_acquire(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a acquire, align 16, !dbg !7
  ret i128 %0, !dbg !7
}
; CHECK-LABEL: atomic128_load_acquire
; CHECK: call i128 @__tsan_atomic128_load(ptr %a, i32 2), !dbg

define i128 @atomic128_load_seq_cst(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i128, ptr %a seq_cst, align 16, !dbg !7
  ret i128 %0, !dbg !7
}
; CHECK-LABEL: atomic128_load_seq_cst
; CHECK: call i128 @__tsan_atomic128_load(ptr %a, i32 5), !dbg

define void @atomic128_store_unordered(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a unordered, align 16, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_store_unordered
; CHECK: call void @__tsan_atomic128_store(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_store_monotonic(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a monotonic, align 16, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_store_monotonic
; CHECK: call void @__tsan_atomic128_store(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_store_release(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a release, align 16, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_store_release
; CHECK: call void @__tsan_atomic128_store(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_store_seq_cst(ptr %a) nounwind uwtable {
entry:
  store atomic i128 0, ptr %a seq_cst, align 16, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_store_seq_cst
; CHECK: call void @__tsan_atomic128_store(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_xchg_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xchg_monotonic
; CHECK: call i128 @__tsan_atomic128_exchange(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_add_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_add_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_add(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_sub_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_sub_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_sub(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_and_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_and_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_and(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_or_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_or_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_or(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_xor_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xor_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_xor(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_nand_monotonic(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_nand_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_nand(ptr %a, i128 0, i32 0), !dbg

define void @atomic128_xchg_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xchg_acquire
; CHECK: call i128 @__tsan_atomic128_exchange(ptr %a, i128 0, i32 2), !dbg

define void @atomic128_add_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_add_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_add(ptr %a, i128 0, i32 2), !dbg

define void @atomic128_sub_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_sub_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_sub(ptr %a, i128 0, i32 2), !dbg

define void @atomic128_and_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_and_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_and(ptr %a, i128 0, i32 2), !dbg

define void @atomic128_or_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_or_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_or(ptr %a, i128 0, i32 2), !dbg

define void @atomic128_xor_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xor_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_xor(ptr %a, i128 0, i32 2), !dbg

define void @atomic128_nand_acquire(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_nand_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_nand(ptr %a, i128 0, i32 2), !dbg

define void @atomic128_xchg_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xchg_release
; CHECK: call i128 @__tsan_atomic128_exchange(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_add_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_add_release
; CHECK: call i128 @__tsan_atomic128_fetch_add(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_sub_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_sub_release
; CHECK: call i128 @__tsan_atomic128_fetch_sub(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_and_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_and_release
; CHECK: call i128 @__tsan_atomic128_fetch_and(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_or_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_or_release
; CHECK: call i128 @__tsan_atomic128_fetch_or(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_xor_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xor_release
; CHECK: call i128 @__tsan_atomic128_fetch_xor(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_nand_release(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_nand_release
; CHECK: call i128 @__tsan_atomic128_fetch_nand(ptr %a, i128 0, i32 3), !dbg

define void @atomic128_xchg_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xchg_acq_rel
; CHECK: call i128 @__tsan_atomic128_exchange(ptr %a, i128 0, i32 4), !dbg

define void @atomic128_add_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_add_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_add(ptr %a, i128 0, i32 4), !dbg

define void @atomic128_sub_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_sub_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_sub(ptr %a, i128 0, i32 4), !dbg

define void @atomic128_and_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_and_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_and(ptr %a, i128 0, i32 4), !dbg

define void @atomic128_or_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_or_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_or(ptr %a, i128 0, i32 4), !dbg

define void @atomic128_xor_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xor_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_xor(ptr %a, i128 0, i32 4), !dbg

define void @atomic128_nand_acq_rel(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_nand_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_nand(ptr %a, i128 0, i32 4), !dbg

define void @atomic128_xchg_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xchg ptr %a, i128 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xchg_seq_cst
; CHECK: call i128 @__tsan_atomic128_exchange(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_add_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw add ptr %a, i128 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_add_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_add(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_sub_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw sub ptr %a, i128 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_sub_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_sub(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_and_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw and ptr %a, i128 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_and_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_and(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_or_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw or ptr %a, i128 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_or_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_or(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_xor_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw xor ptr %a, i128 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_xor_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_xor(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_nand_seq_cst(ptr %a) nounwind uwtable {
entry:
  atomicrmw nand ptr %a, i128 0 seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_nand_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_nand(ptr %a, i128 0, i32 5), !dbg

define void @atomic128_cas_monotonic(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 monotonic monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_cas_monotonic
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(ptr %a, i128 0, i128 1, i32 0, i32 0), !dbg

define void @atomic128_cas_acquire(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 acquire acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_cas_acquire
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(ptr %a, i128 0, i128 1, i32 2, i32 2), !dbg

define void @atomic128_cas_release(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 release monotonic, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_cas_release
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(ptr %a, i128 0, i128 1, i32 3, i32 0), !dbg

define void @atomic128_cas_acq_rel(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 acq_rel acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_cas_acq_rel
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(ptr %a, i128 0, i128 1, i32 4, i32 2), !dbg

define void @atomic128_cas_seq_cst(ptr %a) nounwind uwtable {
entry:
  cmpxchg ptr %a, i128 0, i128 1 seq_cst seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic128_cas_seq_cst
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(ptr %a, i128 0, i128 1, i32 5, i32 5), !dbg

define void @atomic_signal_fence_acquire() nounwind uwtable {
entry:
  fence syncscope("singlethread") acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_signal_fence_acquire
; CHECK: call void @__tsan_atomic_signal_fence(i32 2), !dbg

define void @atomic_thread_fence_acquire() nounwind uwtable {
entry:
  fence  acquire, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_thread_fence_acquire
; CHECK: call void @__tsan_atomic_thread_fence(i32 2), !dbg

define void @atomic_signal_fence_release() nounwind uwtable {
entry:
  fence syncscope("singlethread") release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_signal_fence_release
; CHECK: call void @__tsan_atomic_signal_fence(i32 3), !dbg

define void @atomic_thread_fence_release() nounwind uwtable {
entry:
  fence  release, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_thread_fence_release
; CHECK: call void @__tsan_atomic_thread_fence(i32 3), !dbg

define void @atomic_signal_fence_acq_rel() nounwind uwtable {
entry:
  fence syncscope("singlethread") acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_signal_fence_acq_rel
; CHECK: call void @__tsan_atomic_signal_fence(i32 4), !dbg

define void @atomic_thread_fence_acq_rel() nounwind uwtable {
entry:
  fence  acq_rel, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_thread_fence_acq_rel
; CHECK: call void @__tsan_atomic_thread_fence(i32 4), !dbg

define void @atomic_signal_fence_seq_cst() nounwind uwtable {
entry:
  fence syncscope("singlethread") seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_signal_fence_seq_cst
; CHECK: call void @__tsan_atomic_signal_fence(i32 5), !dbg

define void @atomic_thread_fence_seq_cst() nounwind uwtable {
entry:
  fence  seq_cst, !dbg !7
  ret void, !dbg !7
}
; CHECK-LABEL: atomic_thread_fence_seq_cst
; CHECK: call void @__tsan_atomic_thread_fence(i32 5), !dbg

; CHECK:    declare void @__tsan_atomic32_store(ptr, i32, i32)
; EXT:      declare void @__tsan_atomic32_store(ptr, i32 signext, i32 signext)
; MIPS_EXT: declare void @__tsan_atomic32_store(ptr, i32 signext, i32 signext)
; LA_EXT:   declare void @__tsan_atomic32_store(ptr, i32 signext, i32 signext)

; CHECK:    declare i32 @__tsan_atomic32_compare_exchange_val(ptr, i32, i32, i32, i32)
; EXT:      declare signext i32 @__tsan_atomic32_compare_exchange_val(ptr, i32 signext, i32 signext, i32 signext, i32 signext)
; MIPS_EXT: declare i32 @__tsan_atomic32_compare_exchange_val(ptr, i32 signext, i32 signext, i32 signext, i32 signext)
; LA_EXT:   declare signext i32 @__tsan_atomic32_compare_exchange_val(ptr, i32 signext, i32 signext, i32 signext, i32 signext)

; CHECK:    declare i64 @__tsan_atomic64_load(ptr, i32)
; EXT:      declare i64 @__tsan_atomic64_load(ptr, i32 signext)
; MIPS_EXT: declare i64 @__tsan_atomic64_load(ptr, i32 signext)
; LA_EXT:   declare i64 @__tsan_atomic64_load(ptr, i32 signext)

; CHECK:    declare void @__tsan_atomic64_store(ptr, i64, i32)
; EXT:      declare void @__tsan_atomic64_store(ptr, i64, i32 signext)
; MIPS_EXT: declare void @__tsan_atomic64_store(ptr, i64, i32 signext)
; LA_EXT:   declare void @__tsan_atomic64_store(ptr, i64, i32 signext)

; CHECK:    declare i64 @__tsan_atomic64_fetch_add(ptr, i64, i32)
; EXT:      declare i64 @__tsan_atomic64_fetch_add(ptr, i64, i32 signext)
; MIPS_EXT: declare i64 @__tsan_atomic64_fetch_add(ptr, i64, i32 signext)
; LA_EXT:   declare i64 @__tsan_atomic64_fetch_add(ptr, i64, i32 signext)

; CHECK:    declare i64 @__tsan_atomic64_compare_exchange_val(ptr, i64, i64, i32, i32)
; EXT:      declare i64 @__tsan_atomic64_compare_exchange_val(ptr, i64, i64, i32 signext, i32 signext)
; MIPS_EXT: declare i64 @__tsan_atomic64_compare_exchange_val(ptr, i64, i64, i32 signext, i32 signext)
; LA_EXT:   declare i64 @__tsan_atomic64_compare_exchange_val(ptr, i64, i64, i32 signext, i32 signext)

; CHECK:    declare void @__tsan_atomic_thread_fence(i32)
; EXT:      declare void @__tsan_atomic_thread_fence(i32 signext)
; MIPS_EXT: declare void @__tsan_atomic_thread_fence(i32 signext)
; LA_EXT:   declare void @__tsan_atomic_thread_fence(i32 signext)

; CHECK:    declare void @__tsan_atomic_signal_fence(i32)
; EXT:      declare void @__tsan_atomic_signal_fence(i32 signext)
; MIPS_EXT: declare void @__tsan_atomic_signal_fence(i32 signext)
; LA_EXT:   declare void @__tsan_atomic_signal_fence(i32 signext)

; CHECK:    declare ptr @__tsan_memset(ptr, i32, i64)
; EXT:      declare ptr @__tsan_memset(ptr, i32 signext, i64)
; MIPS_EXT: declare ptr @__tsan_memset(ptr, i32 signext, i64)
; LA_EXT:   declare ptr @__tsan_memset(ptr, i32 signext, i64)

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!8}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"PIC Level", i32 2}

!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = !DIFile(filename: "atomic.cpp", directory: "/tmp")
!6 = distinct !DISubprogram(name: "test", scope: !5, file: !5, line: 99, type: !4, isLocal: false, isDefinition: true, scopeLine: 100, flags: DIFlagPrototyped, isOptimized: false, unit: !8, retainedNodes: !3)
!7 = !DILocation(line: 100, column: 1, scope: !6)

!8 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !5,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
