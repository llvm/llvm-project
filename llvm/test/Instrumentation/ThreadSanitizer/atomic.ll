; RUN: opt < %s -tsan -S | FileCheck %s
; Check that atomic memory operations are converted to calls into ThreadSanitizer runtime.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i8 @atomic8_load_unordered(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a unordered, align 1
  ret i8 %0
}
; CHECK: atomic8_load_unordered
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 0)

define i8 @atomic8_load_monotonic(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a monotonic, align 1
  ret i8 %0
}
; CHECK: atomic8_load_monotonic
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 0)

define i8 @atomic8_load_acquire(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a acquire, align 1
  ret i8 %0
}
; CHECK: atomic8_load_acquire
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 2)

define i8 @atomic8_load_seq_cst(i8* %a) nounwind uwtable {
entry:
  %0 = load atomic i8* %a seq_cst, align 1
  ret i8 %0
}
; CHECK: atomic8_load_seq_cst
; CHECK: call i8 @__tsan_atomic8_load(i8* %a, i32 5)

define void @atomic8_store_unordered(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a unordered, align 1
  ret void
}
; CHECK: atomic8_store_unordered
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 0)

define void @atomic8_store_monotonic(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a monotonic, align 1
  ret void
}
; CHECK: atomic8_store_monotonic
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 0)

define void @atomic8_store_release(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a release, align 1
  ret void
}
; CHECK: atomic8_store_release
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 3)

define void @atomic8_store_seq_cst(i8* %a) nounwind uwtable {
entry:
  store atomic i8 0, i8* %a seq_cst, align 1
  ret void
}
; CHECK: atomic8_store_seq_cst
; CHECK: call void @__tsan_atomic8_store(i8* %a, i8 0, i32 5)

define void @atomic8_xchg_monotonic(i8* %a) nounwind uwtable {
entry:
  atomicrmw xchg i8* %a, i8 0 monotonic
  ret void
}
; CHECK: atomic8_xchg_monotonic
; CHECK: call i8 @__tsan_atomic8_exchange(i8* %a, i8 0, i32 0)

define void @atomic8_add_monotonic(i8* %a) nounwind uwtable {
entry:
  atomicrmw add i8* %a, i8 0 monotonic
  ret void
}
; CHECK: atomic8_add_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_add(i8* %a, i8 0, i32 0)

define void @atomic8_sub_monotonic(i8* %a) nounwind uwtable {
entry:
  atomicrmw sub i8* %a, i8 0 monotonic
  ret void
}
; CHECK: atomic8_sub_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_sub(i8* %a, i8 0, i32 0)

define void @atomic8_and_monotonic(i8* %a) nounwind uwtable {
entry:
  atomicrmw and i8* %a, i8 0 monotonic
  ret void
}
; CHECK: atomic8_and_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_and(i8* %a, i8 0, i32 0)

define void @atomic8_or_monotonic(i8* %a) nounwind uwtable {
entry:
  atomicrmw or i8* %a, i8 0 monotonic
  ret void
}
; CHECK: atomic8_or_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_or(i8* %a, i8 0, i32 0)

define void @atomic8_xor_monotonic(i8* %a) nounwind uwtable {
entry:
  atomicrmw xor i8* %a, i8 0 monotonic
  ret void
}
; CHECK: atomic8_xor_monotonic
; CHECK: call i8 @__tsan_atomic8_fetch_xor(i8* %a, i8 0, i32 0)

define void @atomic8_xchg_acquire(i8* %a) nounwind uwtable {
entry:
  atomicrmw xchg i8* %a, i8 0 acquire
  ret void
}
; CHECK: atomic8_xchg_acquire
; CHECK: call i8 @__tsan_atomic8_exchange(i8* %a, i8 0, i32 2)

define void @atomic8_add_acquire(i8* %a) nounwind uwtable {
entry:
  atomicrmw add i8* %a, i8 0 acquire
  ret void
}
; CHECK: atomic8_add_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_add(i8* %a, i8 0, i32 2)

define void @atomic8_sub_acquire(i8* %a) nounwind uwtable {
entry:
  atomicrmw sub i8* %a, i8 0 acquire
  ret void
}
; CHECK: atomic8_sub_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_sub(i8* %a, i8 0, i32 2)

define void @atomic8_and_acquire(i8* %a) nounwind uwtable {
entry:
  atomicrmw and i8* %a, i8 0 acquire
  ret void
}
; CHECK: atomic8_and_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_and(i8* %a, i8 0, i32 2)

define void @atomic8_or_acquire(i8* %a) nounwind uwtable {
entry:
  atomicrmw or i8* %a, i8 0 acquire
  ret void
}
; CHECK: atomic8_or_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_or(i8* %a, i8 0, i32 2)

define void @atomic8_xor_acquire(i8* %a) nounwind uwtable {
entry:
  atomicrmw xor i8* %a, i8 0 acquire
  ret void
}
; CHECK: atomic8_xor_acquire
; CHECK: call i8 @__tsan_atomic8_fetch_xor(i8* %a, i8 0, i32 2)

define void @atomic8_xchg_release(i8* %a) nounwind uwtable {
entry:
  atomicrmw xchg i8* %a, i8 0 release
  ret void
}
; CHECK: atomic8_xchg_release
; CHECK: call i8 @__tsan_atomic8_exchange(i8* %a, i8 0, i32 3)

define void @atomic8_add_release(i8* %a) nounwind uwtable {
entry:
  atomicrmw add i8* %a, i8 0 release
  ret void
}
; CHECK: atomic8_add_release
; CHECK: call i8 @__tsan_atomic8_fetch_add(i8* %a, i8 0, i32 3)

define void @atomic8_sub_release(i8* %a) nounwind uwtable {
entry:
  atomicrmw sub i8* %a, i8 0 release
  ret void
}
; CHECK: atomic8_sub_release
; CHECK: call i8 @__tsan_atomic8_fetch_sub(i8* %a, i8 0, i32 3)

define void @atomic8_and_release(i8* %a) nounwind uwtable {
entry:
  atomicrmw and i8* %a, i8 0 release
  ret void
}
; CHECK: atomic8_and_release
; CHECK: call i8 @__tsan_atomic8_fetch_and(i8* %a, i8 0, i32 3)

define void @atomic8_or_release(i8* %a) nounwind uwtable {
entry:
  atomicrmw or i8* %a, i8 0 release
  ret void
}
; CHECK: atomic8_or_release
; CHECK: call i8 @__tsan_atomic8_fetch_or(i8* %a, i8 0, i32 3)

define void @atomic8_xor_release(i8* %a) nounwind uwtable {
entry:
  atomicrmw xor i8* %a, i8 0 release
  ret void
}
; CHECK: atomic8_xor_release
; CHECK: call i8 @__tsan_atomic8_fetch_xor(i8* %a, i8 0, i32 3)

define void @atomic8_xchg_acq_rel(i8* %a) nounwind uwtable {
entry:
  atomicrmw xchg i8* %a, i8 0 acq_rel
  ret void
}
; CHECK: atomic8_xchg_acq_rel
; CHECK: call i8 @__tsan_atomic8_exchange(i8* %a, i8 0, i32 4)

define void @atomic8_add_acq_rel(i8* %a) nounwind uwtable {
entry:
  atomicrmw add i8* %a, i8 0 acq_rel
  ret void
}
; CHECK: atomic8_add_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_add(i8* %a, i8 0, i32 4)

define void @atomic8_sub_acq_rel(i8* %a) nounwind uwtable {
entry:
  atomicrmw sub i8* %a, i8 0 acq_rel
  ret void
}
; CHECK: atomic8_sub_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_sub(i8* %a, i8 0, i32 4)

define void @atomic8_and_acq_rel(i8* %a) nounwind uwtable {
entry:
  atomicrmw and i8* %a, i8 0 acq_rel
  ret void
}
; CHECK: atomic8_and_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_and(i8* %a, i8 0, i32 4)

define void @atomic8_or_acq_rel(i8* %a) nounwind uwtable {
entry:
  atomicrmw or i8* %a, i8 0 acq_rel
  ret void
}
; CHECK: atomic8_or_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_or(i8* %a, i8 0, i32 4)

define void @atomic8_xor_acq_rel(i8* %a) nounwind uwtable {
entry:
  atomicrmw xor i8* %a, i8 0 acq_rel
  ret void
}
; CHECK: atomic8_xor_acq_rel
; CHECK: call i8 @__tsan_atomic8_fetch_xor(i8* %a, i8 0, i32 4)

define void @atomic8_xchg_seq_cst(i8* %a) nounwind uwtable {
entry:
  atomicrmw xchg i8* %a, i8 0 seq_cst
  ret void
}
; CHECK: atomic8_xchg_seq_cst
; CHECK: call i8 @__tsan_atomic8_exchange(i8* %a, i8 0, i32 5)

define void @atomic8_add_seq_cst(i8* %a) nounwind uwtable {
entry:
  atomicrmw add i8* %a, i8 0 seq_cst
  ret void
}
; CHECK: atomic8_add_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_add(i8* %a, i8 0, i32 5)

define void @atomic8_sub_seq_cst(i8* %a) nounwind uwtable {
entry:
  atomicrmw sub i8* %a, i8 0 seq_cst
  ret void
}
; CHECK: atomic8_sub_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_sub(i8* %a, i8 0, i32 5)

define void @atomic8_and_seq_cst(i8* %a) nounwind uwtable {
entry:
  atomicrmw and i8* %a, i8 0 seq_cst
  ret void
}
; CHECK: atomic8_and_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_and(i8* %a, i8 0, i32 5)

define void @atomic8_or_seq_cst(i8* %a) nounwind uwtable {
entry:
  atomicrmw or i8* %a, i8 0 seq_cst
  ret void
}
; CHECK: atomic8_or_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_or(i8* %a, i8 0, i32 5)

define void @atomic8_xor_seq_cst(i8* %a) nounwind uwtable {
entry:
  atomicrmw xor i8* %a, i8 0 seq_cst
  ret void
}
; CHECK: atomic8_xor_seq_cst
; CHECK: call i8 @__tsan_atomic8_fetch_xor(i8* %a, i8 0, i32 5)

define void @atomic8_cas_monotonic(i8* %a) nounwind uwtable {
entry:
  cmpxchg i8* %a, i8 0, i8 1 monotonic
  ret void
}
; CHECK: atomic8_cas_monotonic
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(i8* %a, i8 0, i8 1, i32 0)

define void @atomic8_cas_acquire(i8* %a) nounwind uwtable {
entry:
  cmpxchg i8* %a, i8 0, i8 1 acquire
  ret void
}
; CHECK: atomic8_cas_acquire
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(i8* %a, i8 0, i8 1, i32 2)

define void @atomic8_cas_release(i8* %a) nounwind uwtable {
entry:
  cmpxchg i8* %a, i8 0, i8 1 release
  ret void
}
; CHECK: atomic8_cas_release
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(i8* %a, i8 0, i8 1, i32 3)

define void @atomic8_cas_acq_rel(i8* %a) nounwind uwtable {
entry:
  cmpxchg i8* %a, i8 0, i8 1 acq_rel
  ret void
}
; CHECK: atomic8_cas_acq_rel
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(i8* %a, i8 0, i8 1, i32 4)

define void @atomic8_cas_seq_cst(i8* %a) nounwind uwtable {
entry:
  cmpxchg i8* %a, i8 0, i8 1 seq_cst
  ret void
}
; CHECK: atomic8_cas_seq_cst
; CHECK: call i8 @__tsan_atomic8_compare_exchange_val(i8* %a, i8 0, i8 1, i32 5)

define i16 @atomic16_load_unordered(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a unordered, align 2
  ret i16 %0
}
; CHECK: atomic16_load_unordered
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 0)

define i16 @atomic16_load_monotonic(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a monotonic, align 2
  ret i16 %0
}
; CHECK: atomic16_load_monotonic
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 0)

define i16 @atomic16_load_acquire(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a acquire, align 2
  ret i16 %0
}
; CHECK: atomic16_load_acquire
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 2)

define i16 @atomic16_load_seq_cst(i16* %a) nounwind uwtable {
entry:
  %0 = load atomic i16* %a seq_cst, align 2
  ret i16 %0
}
; CHECK: atomic16_load_seq_cst
; CHECK: call i16 @__tsan_atomic16_load(i16* %a, i32 5)

define void @atomic16_store_unordered(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a unordered, align 2
  ret void
}
; CHECK: atomic16_store_unordered
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 0)

define void @atomic16_store_monotonic(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a monotonic, align 2
  ret void
}
; CHECK: atomic16_store_monotonic
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 0)

define void @atomic16_store_release(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a release, align 2
  ret void
}
; CHECK: atomic16_store_release
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 3)

define void @atomic16_store_seq_cst(i16* %a) nounwind uwtable {
entry:
  store atomic i16 0, i16* %a seq_cst, align 2
  ret void
}
; CHECK: atomic16_store_seq_cst
; CHECK: call void @__tsan_atomic16_store(i16* %a, i16 0, i32 5)

define void @atomic16_xchg_monotonic(i16* %a) nounwind uwtable {
entry:
  atomicrmw xchg i16* %a, i16 0 monotonic
  ret void
}
; CHECK: atomic16_xchg_monotonic
; CHECK: call i16 @__tsan_atomic16_exchange(i16* %a, i16 0, i32 0)

define void @atomic16_add_monotonic(i16* %a) nounwind uwtable {
entry:
  atomicrmw add i16* %a, i16 0 monotonic
  ret void
}
; CHECK: atomic16_add_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_add(i16* %a, i16 0, i32 0)

define void @atomic16_sub_monotonic(i16* %a) nounwind uwtable {
entry:
  atomicrmw sub i16* %a, i16 0 monotonic
  ret void
}
; CHECK: atomic16_sub_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_sub(i16* %a, i16 0, i32 0)

define void @atomic16_and_monotonic(i16* %a) nounwind uwtable {
entry:
  atomicrmw and i16* %a, i16 0 monotonic
  ret void
}
; CHECK: atomic16_and_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_and(i16* %a, i16 0, i32 0)

define void @atomic16_or_monotonic(i16* %a) nounwind uwtable {
entry:
  atomicrmw or i16* %a, i16 0 monotonic
  ret void
}
; CHECK: atomic16_or_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_or(i16* %a, i16 0, i32 0)

define void @atomic16_xor_monotonic(i16* %a) nounwind uwtable {
entry:
  atomicrmw xor i16* %a, i16 0 monotonic
  ret void
}
; CHECK: atomic16_xor_monotonic
; CHECK: call i16 @__tsan_atomic16_fetch_xor(i16* %a, i16 0, i32 0)

define void @atomic16_xchg_acquire(i16* %a) nounwind uwtable {
entry:
  atomicrmw xchg i16* %a, i16 0 acquire
  ret void
}
; CHECK: atomic16_xchg_acquire
; CHECK: call i16 @__tsan_atomic16_exchange(i16* %a, i16 0, i32 2)

define void @atomic16_add_acquire(i16* %a) nounwind uwtable {
entry:
  atomicrmw add i16* %a, i16 0 acquire
  ret void
}
; CHECK: atomic16_add_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_add(i16* %a, i16 0, i32 2)

define void @atomic16_sub_acquire(i16* %a) nounwind uwtable {
entry:
  atomicrmw sub i16* %a, i16 0 acquire
  ret void
}
; CHECK: atomic16_sub_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_sub(i16* %a, i16 0, i32 2)

define void @atomic16_and_acquire(i16* %a) nounwind uwtable {
entry:
  atomicrmw and i16* %a, i16 0 acquire
  ret void
}
; CHECK: atomic16_and_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_and(i16* %a, i16 0, i32 2)

define void @atomic16_or_acquire(i16* %a) nounwind uwtable {
entry:
  atomicrmw or i16* %a, i16 0 acquire
  ret void
}
; CHECK: atomic16_or_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_or(i16* %a, i16 0, i32 2)

define void @atomic16_xor_acquire(i16* %a) nounwind uwtable {
entry:
  atomicrmw xor i16* %a, i16 0 acquire
  ret void
}
; CHECK: atomic16_xor_acquire
; CHECK: call i16 @__tsan_atomic16_fetch_xor(i16* %a, i16 0, i32 2)

define void @atomic16_xchg_release(i16* %a) nounwind uwtable {
entry:
  atomicrmw xchg i16* %a, i16 0 release
  ret void
}
; CHECK: atomic16_xchg_release
; CHECK: call i16 @__tsan_atomic16_exchange(i16* %a, i16 0, i32 3)

define void @atomic16_add_release(i16* %a) nounwind uwtable {
entry:
  atomicrmw add i16* %a, i16 0 release
  ret void
}
; CHECK: atomic16_add_release
; CHECK: call i16 @__tsan_atomic16_fetch_add(i16* %a, i16 0, i32 3)

define void @atomic16_sub_release(i16* %a) nounwind uwtable {
entry:
  atomicrmw sub i16* %a, i16 0 release
  ret void
}
; CHECK: atomic16_sub_release
; CHECK: call i16 @__tsan_atomic16_fetch_sub(i16* %a, i16 0, i32 3)

define void @atomic16_and_release(i16* %a) nounwind uwtable {
entry:
  atomicrmw and i16* %a, i16 0 release
  ret void
}
; CHECK: atomic16_and_release
; CHECK: call i16 @__tsan_atomic16_fetch_and(i16* %a, i16 0, i32 3)

define void @atomic16_or_release(i16* %a) nounwind uwtable {
entry:
  atomicrmw or i16* %a, i16 0 release
  ret void
}
; CHECK: atomic16_or_release
; CHECK: call i16 @__tsan_atomic16_fetch_or(i16* %a, i16 0, i32 3)

define void @atomic16_xor_release(i16* %a) nounwind uwtable {
entry:
  atomicrmw xor i16* %a, i16 0 release
  ret void
}
; CHECK: atomic16_xor_release
; CHECK: call i16 @__tsan_atomic16_fetch_xor(i16* %a, i16 0, i32 3)

define void @atomic16_xchg_acq_rel(i16* %a) nounwind uwtable {
entry:
  atomicrmw xchg i16* %a, i16 0 acq_rel
  ret void
}
; CHECK: atomic16_xchg_acq_rel
; CHECK: call i16 @__tsan_atomic16_exchange(i16* %a, i16 0, i32 4)

define void @atomic16_add_acq_rel(i16* %a) nounwind uwtable {
entry:
  atomicrmw add i16* %a, i16 0 acq_rel
  ret void
}
; CHECK: atomic16_add_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_add(i16* %a, i16 0, i32 4)

define void @atomic16_sub_acq_rel(i16* %a) nounwind uwtable {
entry:
  atomicrmw sub i16* %a, i16 0 acq_rel
  ret void
}
; CHECK: atomic16_sub_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_sub(i16* %a, i16 0, i32 4)

define void @atomic16_and_acq_rel(i16* %a) nounwind uwtable {
entry:
  atomicrmw and i16* %a, i16 0 acq_rel
  ret void
}
; CHECK: atomic16_and_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_and(i16* %a, i16 0, i32 4)

define void @atomic16_or_acq_rel(i16* %a) nounwind uwtable {
entry:
  atomicrmw or i16* %a, i16 0 acq_rel
  ret void
}
; CHECK: atomic16_or_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_or(i16* %a, i16 0, i32 4)

define void @atomic16_xor_acq_rel(i16* %a) nounwind uwtable {
entry:
  atomicrmw xor i16* %a, i16 0 acq_rel
  ret void
}
; CHECK: atomic16_xor_acq_rel
; CHECK: call i16 @__tsan_atomic16_fetch_xor(i16* %a, i16 0, i32 4)

define void @atomic16_xchg_seq_cst(i16* %a) nounwind uwtable {
entry:
  atomicrmw xchg i16* %a, i16 0 seq_cst
  ret void
}
; CHECK: atomic16_xchg_seq_cst
; CHECK: call i16 @__tsan_atomic16_exchange(i16* %a, i16 0, i32 5)

define void @atomic16_add_seq_cst(i16* %a) nounwind uwtable {
entry:
  atomicrmw add i16* %a, i16 0 seq_cst
  ret void
}
; CHECK: atomic16_add_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_add(i16* %a, i16 0, i32 5)

define void @atomic16_sub_seq_cst(i16* %a) nounwind uwtable {
entry:
  atomicrmw sub i16* %a, i16 0 seq_cst
  ret void
}
; CHECK: atomic16_sub_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_sub(i16* %a, i16 0, i32 5)

define void @atomic16_and_seq_cst(i16* %a) nounwind uwtable {
entry:
  atomicrmw and i16* %a, i16 0 seq_cst
  ret void
}
; CHECK: atomic16_and_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_and(i16* %a, i16 0, i32 5)

define void @atomic16_or_seq_cst(i16* %a) nounwind uwtable {
entry:
  atomicrmw or i16* %a, i16 0 seq_cst
  ret void
}
; CHECK: atomic16_or_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_or(i16* %a, i16 0, i32 5)

define void @atomic16_xor_seq_cst(i16* %a) nounwind uwtable {
entry:
  atomicrmw xor i16* %a, i16 0 seq_cst
  ret void
}
; CHECK: atomic16_xor_seq_cst
; CHECK: call i16 @__tsan_atomic16_fetch_xor(i16* %a, i16 0, i32 5)

define void @atomic16_cas_monotonic(i16* %a) nounwind uwtable {
entry:
  cmpxchg i16* %a, i16 0, i16 1 monotonic
  ret void
}
; CHECK: atomic16_cas_monotonic
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(i16* %a, i16 0, i16 1, i32 0)

define void @atomic16_cas_acquire(i16* %a) nounwind uwtable {
entry:
  cmpxchg i16* %a, i16 0, i16 1 acquire
  ret void
}
; CHECK: atomic16_cas_acquire
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(i16* %a, i16 0, i16 1, i32 2)

define void @atomic16_cas_release(i16* %a) nounwind uwtable {
entry:
  cmpxchg i16* %a, i16 0, i16 1 release
  ret void
}
; CHECK: atomic16_cas_release
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(i16* %a, i16 0, i16 1, i32 3)

define void @atomic16_cas_acq_rel(i16* %a) nounwind uwtable {
entry:
  cmpxchg i16* %a, i16 0, i16 1 acq_rel
  ret void
}
; CHECK: atomic16_cas_acq_rel
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(i16* %a, i16 0, i16 1, i32 4)

define void @atomic16_cas_seq_cst(i16* %a) nounwind uwtable {
entry:
  cmpxchg i16* %a, i16 0, i16 1 seq_cst
  ret void
}
; CHECK: atomic16_cas_seq_cst
; CHECK: call i16 @__tsan_atomic16_compare_exchange_val(i16* %a, i16 0, i16 1, i32 5)

define i32 @atomic32_load_unordered(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a unordered, align 4
  ret i32 %0
}
; CHECK: atomic32_load_unordered
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 0)

define i32 @atomic32_load_monotonic(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a monotonic, align 4
  ret i32 %0
}
; CHECK: atomic32_load_monotonic
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 0)

define i32 @atomic32_load_acquire(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a acquire, align 4
  ret i32 %0
}
; CHECK: atomic32_load_acquire
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 2)

define i32 @atomic32_load_seq_cst(i32* %a) nounwind uwtable {
entry:
  %0 = load atomic i32* %a seq_cst, align 4
  ret i32 %0
}
; CHECK: atomic32_load_seq_cst
; CHECK: call i32 @__tsan_atomic32_load(i32* %a, i32 5)

define void @atomic32_store_unordered(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a unordered, align 4
  ret void
}
; CHECK: atomic32_store_unordered
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 0)

define void @atomic32_store_monotonic(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a monotonic, align 4
  ret void
}
; CHECK: atomic32_store_monotonic
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 0)

define void @atomic32_store_release(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a release, align 4
  ret void
}
; CHECK: atomic32_store_release
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 3)

define void @atomic32_store_seq_cst(i32* %a) nounwind uwtable {
entry:
  store atomic i32 0, i32* %a seq_cst, align 4
  ret void
}
; CHECK: atomic32_store_seq_cst
; CHECK: call void @__tsan_atomic32_store(i32* %a, i32 0, i32 5)

define void @atomic32_xchg_monotonic(i32* %a) nounwind uwtable {
entry:
  atomicrmw xchg i32* %a, i32 0 monotonic
  ret void
}
; CHECK: atomic32_xchg_monotonic
; CHECK: call i32 @__tsan_atomic32_exchange(i32* %a, i32 0, i32 0)

define void @atomic32_add_monotonic(i32* %a) nounwind uwtable {
entry:
  atomicrmw add i32* %a, i32 0 monotonic
  ret void
}
; CHECK: atomic32_add_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_add(i32* %a, i32 0, i32 0)

define void @atomic32_sub_monotonic(i32* %a) nounwind uwtable {
entry:
  atomicrmw sub i32* %a, i32 0 monotonic
  ret void
}
; CHECK: atomic32_sub_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_sub(i32* %a, i32 0, i32 0)

define void @atomic32_and_monotonic(i32* %a) nounwind uwtable {
entry:
  atomicrmw and i32* %a, i32 0 monotonic
  ret void
}
; CHECK: atomic32_and_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_and(i32* %a, i32 0, i32 0)

define void @atomic32_or_monotonic(i32* %a) nounwind uwtable {
entry:
  atomicrmw or i32* %a, i32 0 monotonic
  ret void
}
; CHECK: atomic32_or_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_or(i32* %a, i32 0, i32 0)

define void @atomic32_xor_monotonic(i32* %a) nounwind uwtable {
entry:
  atomicrmw xor i32* %a, i32 0 monotonic
  ret void
}
; CHECK: atomic32_xor_monotonic
; CHECK: call i32 @__tsan_atomic32_fetch_xor(i32* %a, i32 0, i32 0)

define void @atomic32_xchg_acquire(i32* %a) nounwind uwtable {
entry:
  atomicrmw xchg i32* %a, i32 0 acquire
  ret void
}
; CHECK: atomic32_xchg_acquire
; CHECK: call i32 @__tsan_atomic32_exchange(i32* %a, i32 0, i32 2)

define void @atomic32_add_acquire(i32* %a) nounwind uwtable {
entry:
  atomicrmw add i32* %a, i32 0 acquire
  ret void
}
; CHECK: atomic32_add_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_add(i32* %a, i32 0, i32 2)

define void @atomic32_sub_acquire(i32* %a) nounwind uwtable {
entry:
  atomicrmw sub i32* %a, i32 0 acquire
  ret void
}
; CHECK: atomic32_sub_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_sub(i32* %a, i32 0, i32 2)

define void @atomic32_and_acquire(i32* %a) nounwind uwtable {
entry:
  atomicrmw and i32* %a, i32 0 acquire
  ret void
}
; CHECK: atomic32_and_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_and(i32* %a, i32 0, i32 2)

define void @atomic32_or_acquire(i32* %a) nounwind uwtable {
entry:
  atomicrmw or i32* %a, i32 0 acquire
  ret void
}
; CHECK: atomic32_or_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_or(i32* %a, i32 0, i32 2)

define void @atomic32_xor_acquire(i32* %a) nounwind uwtable {
entry:
  atomicrmw xor i32* %a, i32 0 acquire
  ret void
}
; CHECK: atomic32_xor_acquire
; CHECK: call i32 @__tsan_atomic32_fetch_xor(i32* %a, i32 0, i32 2)

define void @atomic32_xchg_release(i32* %a) nounwind uwtable {
entry:
  atomicrmw xchg i32* %a, i32 0 release
  ret void
}
; CHECK: atomic32_xchg_release
; CHECK: call i32 @__tsan_atomic32_exchange(i32* %a, i32 0, i32 3)

define void @atomic32_add_release(i32* %a) nounwind uwtable {
entry:
  atomicrmw add i32* %a, i32 0 release
  ret void
}
; CHECK: atomic32_add_release
; CHECK: call i32 @__tsan_atomic32_fetch_add(i32* %a, i32 0, i32 3)

define void @atomic32_sub_release(i32* %a) nounwind uwtable {
entry:
  atomicrmw sub i32* %a, i32 0 release
  ret void
}
; CHECK: atomic32_sub_release
; CHECK: call i32 @__tsan_atomic32_fetch_sub(i32* %a, i32 0, i32 3)

define void @atomic32_and_release(i32* %a) nounwind uwtable {
entry:
  atomicrmw and i32* %a, i32 0 release
  ret void
}
; CHECK: atomic32_and_release
; CHECK: call i32 @__tsan_atomic32_fetch_and(i32* %a, i32 0, i32 3)

define void @atomic32_or_release(i32* %a) nounwind uwtable {
entry:
  atomicrmw or i32* %a, i32 0 release
  ret void
}
; CHECK: atomic32_or_release
; CHECK: call i32 @__tsan_atomic32_fetch_or(i32* %a, i32 0, i32 3)

define void @atomic32_xor_release(i32* %a) nounwind uwtable {
entry:
  atomicrmw xor i32* %a, i32 0 release
  ret void
}
; CHECK: atomic32_xor_release
; CHECK: call i32 @__tsan_atomic32_fetch_xor(i32* %a, i32 0, i32 3)

define void @atomic32_xchg_acq_rel(i32* %a) nounwind uwtable {
entry:
  atomicrmw xchg i32* %a, i32 0 acq_rel
  ret void
}
; CHECK: atomic32_xchg_acq_rel
; CHECK: call i32 @__tsan_atomic32_exchange(i32* %a, i32 0, i32 4)

define void @atomic32_add_acq_rel(i32* %a) nounwind uwtable {
entry:
  atomicrmw add i32* %a, i32 0 acq_rel
  ret void
}
; CHECK: atomic32_add_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_add(i32* %a, i32 0, i32 4)

define void @atomic32_sub_acq_rel(i32* %a) nounwind uwtable {
entry:
  atomicrmw sub i32* %a, i32 0 acq_rel
  ret void
}
; CHECK: atomic32_sub_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_sub(i32* %a, i32 0, i32 4)

define void @atomic32_and_acq_rel(i32* %a) nounwind uwtable {
entry:
  atomicrmw and i32* %a, i32 0 acq_rel
  ret void
}
; CHECK: atomic32_and_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_and(i32* %a, i32 0, i32 4)

define void @atomic32_or_acq_rel(i32* %a) nounwind uwtable {
entry:
  atomicrmw or i32* %a, i32 0 acq_rel
  ret void
}
; CHECK: atomic32_or_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_or(i32* %a, i32 0, i32 4)

define void @atomic32_xor_acq_rel(i32* %a) nounwind uwtable {
entry:
  atomicrmw xor i32* %a, i32 0 acq_rel
  ret void
}
; CHECK: atomic32_xor_acq_rel
; CHECK: call i32 @__tsan_atomic32_fetch_xor(i32* %a, i32 0, i32 4)

define void @atomic32_xchg_seq_cst(i32* %a) nounwind uwtable {
entry:
  atomicrmw xchg i32* %a, i32 0 seq_cst
  ret void
}
; CHECK: atomic32_xchg_seq_cst
; CHECK: call i32 @__tsan_atomic32_exchange(i32* %a, i32 0, i32 5)

define void @atomic32_add_seq_cst(i32* %a) nounwind uwtable {
entry:
  atomicrmw add i32* %a, i32 0 seq_cst
  ret void
}
; CHECK: atomic32_add_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_add(i32* %a, i32 0, i32 5)

define void @atomic32_sub_seq_cst(i32* %a) nounwind uwtable {
entry:
  atomicrmw sub i32* %a, i32 0 seq_cst
  ret void
}
; CHECK: atomic32_sub_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_sub(i32* %a, i32 0, i32 5)

define void @atomic32_and_seq_cst(i32* %a) nounwind uwtable {
entry:
  atomicrmw and i32* %a, i32 0 seq_cst
  ret void
}
; CHECK: atomic32_and_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_and(i32* %a, i32 0, i32 5)

define void @atomic32_or_seq_cst(i32* %a) nounwind uwtable {
entry:
  atomicrmw or i32* %a, i32 0 seq_cst
  ret void
}
; CHECK: atomic32_or_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_or(i32* %a, i32 0, i32 5)

define void @atomic32_xor_seq_cst(i32* %a) nounwind uwtable {
entry:
  atomicrmw xor i32* %a, i32 0 seq_cst
  ret void
}
; CHECK: atomic32_xor_seq_cst
; CHECK: call i32 @__tsan_atomic32_fetch_xor(i32* %a, i32 0, i32 5)

define void @atomic32_cas_monotonic(i32* %a) nounwind uwtable {
entry:
  cmpxchg i32* %a, i32 0, i32 1 monotonic
  ret void
}
; CHECK: atomic32_cas_monotonic
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(i32* %a, i32 0, i32 1, i32 0)

define void @atomic32_cas_acquire(i32* %a) nounwind uwtable {
entry:
  cmpxchg i32* %a, i32 0, i32 1 acquire
  ret void
}
; CHECK: atomic32_cas_acquire
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(i32* %a, i32 0, i32 1, i32 2)

define void @atomic32_cas_release(i32* %a) nounwind uwtable {
entry:
  cmpxchg i32* %a, i32 0, i32 1 release
  ret void
}
; CHECK: atomic32_cas_release
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(i32* %a, i32 0, i32 1, i32 3)

define void @atomic32_cas_acq_rel(i32* %a) nounwind uwtable {
entry:
  cmpxchg i32* %a, i32 0, i32 1 acq_rel
  ret void
}
; CHECK: atomic32_cas_acq_rel
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(i32* %a, i32 0, i32 1, i32 4)

define void @atomic32_cas_seq_cst(i32* %a) nounwind uwtable {
entry:
  cmpxchg i32* %a, i32 0, i32 1 seq_cst
  ret void
}
; CHECK: atomic32_cas_seq_cst
; CHECK: call i32 @__tsan_atomic32_compare_exchange_val(i32* %a, i32 0, i32 1, i32 5)

define i64 @atomic64_load_unordered(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a unordered, align 8
  ret i64 %0
}
; CHECK: atomic64_load_unordered
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 0)

define i64 @atomic64_load_monotonic(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a monotonic, align 8
  ret i64 %0
}
; CHECK: atomic64_load_monotonic
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 0)

define i64 @atomic64_load_acquire(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a acquire, align 8
  ret i64 %0
}
; CHECK: atomic64_load_acquire
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 2)

define i64 @atomic64_load_seq_cst(i64* %a) nounwind uwtable {
entry:
  %0 = load atomic i64* %a seq_cst, align 8
  ret i64 %0
}
; CHECK: atomic64_load_seq_cst
; CHECK: call i64 @__tsan_atomic64_load(i64* %a, i32 5)

define void @atomic64_store_unordered(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a unordered, align 8
  ret void
}
; CHECK: atomic64_store_unordered
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 0)

define void @atomic64_store_monotonic(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a monotonic, align 8
  ret void
}
; CHECK: atomic64_store_monotonic
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 0)

define void @atomic64_store_release(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a release, align 8
  ret void
}
; CHECK: atomic64_store_release
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 3)

define void @atomic64_store_seq_cst(i64* %a) nounwind uwtable {
entry:
  store atomic i64 0, i64* %a seq_cst, align 8
  ret void
}
; CHECK: atomic64_store_seq_cst
; CHECK: call void @__tsan_atomic64_store(i64* %a, i64 0, i32 5)

define void @atomic64_xchg_monotonic(i64* %a) nounwind uwtable {
entry:
  atomicrmw xchg i64* %a, i64 0 monotonic
  ret void
}
; CHECK: atomic64_xchg_monotonic
; CHECK: call i64 @__tsan_atomic64_exchange(i64* %a, i64 0, i32 0)

define void @atomic64_add_monotonic(i64* %a) nounwind uwtable {
entry:
  atomicrmw add i64* %a, i64 0 monotonic
  ret void
}
; CHECK: atomic64_add_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_add(i64* %a, i64 0, i32 0)

define void @atomic64_sub_monotonic(i64* %a) nounwind uwtable {
entry:
  atomicrmw sub i64* %a, i64 0 monotonic
  ret void
}
; CHECK: atomic64_sub_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_sub(i64* %a, i64 0, i32 0)

define void @atomic64_and_monotonic(i64* %a) nounwind uwtable {
entry:
  atomicrmw and i64* %a, i64 0 monotonic
  ret void
}
; CHECK: atomic64_and_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_and(i64* %a, i64 0, i32 0)

define void @atomic64_or_monotonic(i64* %a) nounwind uwtable {
entry:
  atomicrmw or i64* %a, i64 0 monotonic
  ret void
}
; CHECK: atomic64_or_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_or(i64* %a, i64 0, i32 0)

define void @atomic64_xor_monotonic(i64* %a) nounwind uwtable {
entry:
  atomicrmw xor i64* %a, i64 0 monotonic
  ret void
}
; CHECK: atomic64_xor_monotonic
; CHECK: call i64 @__tsan_atomic64_fetch_xor(i64* %a, i64 0, i32 0)

define void @atomic64_xchg_acquire(i64* %a) nounwind uwtable {
entry:
  atomicrmw xchg i64* %a, i64 0 acquire
  ret void
}
; CHECK: atomic64_xchg_acquire
; CHECK: call i64 @__tsan_atomic64_exchange(i64* %a, i64 0, i32 2)

define void @atomic64_add_acquire(i64* %a) nounwind uwtable {
entry:
  atomicrmw add i64* %a, i64 0 acquire
  ret void
}
; CHECK: atomic64_add_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_add(i64* %a, i64 0, i32 2)

define void @atomic64_sub_acquire(i64* %a) nounwind uwtable {
entry:
  atomicrmw sub i64* %a, i64 0 acquire
  ret void
}
; CHECK: atomic64_sub_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_sub(i64* %a, i64 0, i32 2)

define void @atomic64_and_acquire(i64* %a) nounwind uwtable {
entry:
  atomicrmw and i64* %a, i64 0 acquire
  ret void
}
; CHECK: atomic64_and_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_and(i64* %a, i64 0, i32 2)

define void @atomic64_or_acquire(i64* %a) nounwind uwtable {
entry:
  atomicrmw or i64* %a, i64 0 acquire
  ret void
}
; CHECK: atomic64_or_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_or(i64* %a, i64 0, i32 2)

define void @atomic64_xor_acquire(i64* %a) nounwind uwtable {
entry:
  atomicrmw xor i64* %a, i64 0 acquire
  ret void
}
; CHECK: atomic64_xor_acquire
; CHECK: call i64 @__tsan_atomic64_fetch_xor(i64* %a, i64 0, i32 2)

define void @atomic64_xchg_release(i64* %a) nounwind uwtable {
entry:
  atomicrmw xchg i64* %a, i64 0 release
  ret void
}
; CHECK: atomic64_xchg_release
; CHECK: call i64 @__tsan_atomic64_exchange(i64* %a, i64 0, i32 3)

define void @atomic64_add_release(i64* %a) nounwind uwtable {
entry:
  atomicrmw add i64* %a, i64 0 release
  ret void
}
; CHECK: atomic64_add_release
; CHECK: call i64 @__tsan_atomic64_fetch_add(i64* %a, i64 0, i32 3)

define void @atomic64_sub_release(i64* %a) nounwind uwtable {
entry:
  atomicrmw sub i64* %a, i64 0 release
  ret void
}
; CHECK: atomic64_sub_release
; CHECK: call i64 @__tsan_atomic64_fetch_sub(i64* %a, i64 0, i32 3)

define void @atomic64_and_release(i64* %a) nounwind uwtable {
entry:
  atomicrmw and i64* %a, i64 0 release
  ret void
}
; CHECK: atomic64_and_release
; CHECK: call i64 @__tsan_atomic64_fetch_and(i64* %a, i64 0, i32 3)

define void @atomic64_or_release(i64* %a) nounwind uwtable {
entry:
  atomicrmw or i64* %a, i64 0 release
  ret void
}
; CHECK: atomic64_or_release
; CHECK: call i64 @__tsan_atomic64_fetch_or(i64* %a, i64 0, i32 3)

define void @atomic64_xor_release(i64* %a) nounwind uwtable {
entry:
  atomicrmw xor i64* %a, i64 0 release
  ret void
}
; CHECK: atomic64_xor_release
; CHECK: call i64 @__tsan_atomic64_fetch_xor(i64* %a, i64 0, i32 3)

define void @atomic64_xchg_acq_rel(i64* %a) nounwind uwtable {
entry:
  atomicrmw xchg i64* %a, i64 0 acq_rel
  ret void
}
; CHECK: atomic64_xchg_acq_rel
; CHECK: call i64 @__tsan_atomic64_exchange(i64* %a, i64 0, i32 4)

define void @atomic64_add_acq_rel(i64* %a) nounwind uwtable {
entry:
  atomicrmw add i64* %a, i64 0 acq_rel
  ret void
}
; CHECK: atomic64_add_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_add(i64* %a, i64 0, i32 4)

define void @atomic64_sub_acq_rel(i64* %a) nounwind uwtable {
entry:
  atomicrmw sub i64* %a, i64 0 acq_rel
  ret void
}
; CHECK: atomic64_sub_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_sub(i64* %a, i64 0, i32 4)

define void @atomic64_and_acq_rel(i64* %a) nounwind uwtable {
entry:
  atomicrmw and i64* %a, i64 0 acq_rel
  ret void
}
; CHECK: atomic64_and_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_and(i64* %a, i64 0, i32 4)

define void @atomic64_or_acq_rel(i64* %a) nounwind uwtable {
entry:
  atomicrmw or i64* %a, i64 0 acq_rel
  ret void
}
; CHECK: atomic64_or_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_or(i64* %a, i64 0, i32 4)

define void @atomic64_xor_acq_rel(i64* %a) nounwind uwtable {
entry:
  atomicrmw xor i64* %a, i64 0 acq_rel
  ret void
}
; CHECK: atomic64_xor_acq_rel
; CHECK: call i64 @__tsan_atomic64_fetch_xor(i64* %a, i64 0, i32 4)

define void @atomic64_xchg_seq_cst(i64* %a) nounwind uwtable {
entry:
  atomicrmw xchg i64* %a, i64 0 seq_cst
  ret void
}
; CHECK: atomic64_xchg_seq_cst
; CHECK: call i64 @__tsan_atomic64_exchange(i64* %a, i64 0, i32 5)

define void @atomic64_add_seq_cst(i64* %a) nounwind uwtable {
entry:
  atomicrmw add i64* %a, i64 0 seq_cst
  ret void
}
; CHECK: atomic64_add_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_add(i64* %a, i64 0, i32 5)

define void @atomic64_sub_seq_cst(i64* %a) nounwind uwtable {
entry:
  atomicrmw sub i64* %a, i64 0 seq_cst
  ret void
}
; CHECK: atomic64_sub_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_sub(i64* %a, i64 0, i32 5)

define void @atomic64_and_seq_cst(i64* %a) nounwind uwtable {
entry:
  atomicrmw and i64* %a, i64 0 seq_cst
  ret void
}
; CHECK: atomic64_and_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_and(i64* %a, i64 0, i32 5)

define void @atomic64_or_seq_cst(i64* %a) nounwind uwtable {
entry:
  atomicrmw or i64* %a, i64 0 seq_cst
  ret void
}
; CHECK: atomic64_or_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_or(i64* %a, i64 0, i32 5)

define void @atomic64_xor_seq_cst(i64* %a) nounwind uwtable {
entry:
  atomicrmw xor i64* %a, i64 0 seq_cst
  ret void
}
; CHECK: atomic64_xor_seq_cst
; CHECK: call i64 @__tsan_atomic64_fetch_xor(i64* %a, i64 0, i32 5)

define void @atomic64_cas_monotonic(i64* %a) nounwind uwtable {
entry:
  cmpxchg i64* %a, i64 0, i64 1 monotonic
  ret void
}
; CHECK: atomic64_cas_monotonic
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(i64* %a, i64 0, i64 1, i32 0)

define void @atomic64_cas_acquire(i64* %a) nounwind uwtable {
entry:
  cmpxchg i64* %a, i64 0, i64 1 acquire
  ret void
}
; CHECK: atomic64_cas_acquire
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(i64* %a, i64 0, i64 1, i32 2)

define void @atomic64_cas_release(i64* %a) nounwind uwtable {
entry:
  cmpxchg i64* %a, i64 0, i64 1 release
  ret void
}
; CHECK: atomic64_cas_release
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(i64* %a, i64 0, i64 1, i32 3)

define void @atomic64_cas_acq_rel(i64* %a) nounwind uwtable {
entry:
  cmpxchg i64* %a, i64 0, i64 1 acq_rel
  ret void
}
; CHECK: atomic64_cas_acq_rel
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(i64* %a, i64 0, i64 1, i32 4)

define void @atomic64_cas_seq_cst(i64* %a) nounwind uwtable {
entry:
  cmpxchg i64* %a, i64 0, i64 1 seq_cst
  ret void
}
; CHECK: atomic64_cas_seq_cst
; CHECK: call i64 @__tsan_atomic64_compare_exchange_val(i64* %a, i64 0, i64 1, i32 5)

define i128 @atomic128_load_unordered(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a unordered, align 16
  ret i128 %0
}
; CHECK: atomic128_load_unordered
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 0)

define i128 @atomic128_load_monotonic(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a monotonic, align 16
  ret i128 %0
}
; CHECK: atomic128_load_monotonic
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 0)

define i128 @atomic128_load_acquire(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a acquire, align 16
  ret i128 %0
}
; CHECK: atomic128_load_acquire
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 2)

define i128 @atomic128_load_seq_cst(i128* %a) nounwind uwtable {
entry:
  %0 = load atomic i128* %a seq_cst, align 16
  ret i128 %0
}
; CHECK: atomic128_load_seq_cst
; CHECK: call i128 @__tsan_atomic128_load(i128* %a, i32 5)

define void @atomic128_store_unordered(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a unordered, align 16
  ret void
}
; CHECK: atomic128_store_unordered
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 0)

define void @atomic128_store_monotonic(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a monotonic, align 16
  ret void
}
; CHECK: atomic128_store_monotonic
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 0)

define void @atomic128_store_release(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a release, align 16
  ret void
}
; CHECK: atomic128_store_release
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 3)

define void @atomic128_store_seq_cst(i128* %a) nounwind uwtable {
entry:
  store atomic i128 0, i128* %a seq_cst, align 16
  ret void
}
; CHECK: atomic128_store_seq_cst
; CHECK: call void @__tsan_atomic128_store(i128* %a, i128 0, i32 5)

define void @atomic128_xchg_monotonic(i128* %a) nounwind uwtable {
entry:
  atomicrmw xchg i128* %a, i128 0 monotonic
  ret void
}
; CHECK: atomic128_xchg_monotonic
; CHECK: call i128 @__tsan_atomic128_exchange(i128* %a, i128 0, i32 0)

define void @atomic128_add_monotonic(i128* %a) nounwind uwtable {
entry:
  atomicrmw add i128* %a, i128 0 monotonic
  ret void
}
; CHECK: atomic128_add_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_add(i128* %a, i128 0, i32 0)

define void @atomic128_sub_monotonic(i128* %a) nounwind uwtable {
entry:
  atomicrmw sub i128* %a, i128 0 monotonic
  ret void
}
; CHECK: atomic128_sub_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_sub(i128* %a, i128 0, i32 0)

define void @atomic128_and_monotonic(i128* %a) nounwind uwtable {
entry:
  atomicrmw and i128* %a, i128 0 monotonic
  ret void
}
; CHECK: atomic128_and_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_and(i128* %a, i128 0, i32 0)

define void @atomic128_or_monotonic(i128* %a) nounwind uwtable {
entry:
  atomicrmw or i128* %a, i128 0 monotonic
  ret void
}
; CHECK: atomic128_or_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_or(i128* %a, i128 0, i32 0)

define void @atomic128_xor_monotonic(i128* %a) nounwind uwtable {
entry:
  atomicrmw xor i128* %a, i128 0 monotonic
  ret void
}
; CHECK: atomic128_xor_monotonic
; CHECK: call i128 @__tsan_atomic128_fetch_xor(i128* %a, i128 0, i32 0)

define void @atomic128_xchg_acquire(i128* %a) nounwind uwtable {
entry:
  atomicrmw xchg i128* %a, i128 0 acquire
  ret void
}
; CHECK: atomic128_xchg_acquire
; CHECK: call i128 @__tsan_atomic128_exchange(i128* %a, i128 0, i32 2)

define void @atomic128_add_acquire(i128* %a) nounwind uwtable {
entry:
  atomicrmw add i128* %a, i128 0 acquire
  ret void
}
; CHECK: atomic128_add_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_add(i128* %a, i128 0, i32 2)

define void @atomic128_sub_acquire(i128* %a) nounwind uwtable {
entry:
  atomicrmw sub i128* %a, i128 0 acquire
  ret void
}
; CHECK: atomic128_sub_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_sub(i128* %a, i128 0, i32 2)

define void @atomic128_and_acquire(i128* %a) nounwind uwtable {
entry:
  atomicrmw and i128* %a, i128 0 acquire
  ret void
}
; CHECK: atomic128_and_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_and(i128* %a, i128 0, i32 2)

define void @atomic128_or_acquire(i128* %a) nounwind uwtable {
entry:
  atomicrmw or i128* %a, i128 0 acquire
  ret void
}
; CHECK: atomic128_or_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_or(i128* %a, i128 0, i32 2)

define void @atomic128_xor_acquire(i128* %a) nounwind uwtable {
entry:
  atomicrmw xor i128* %a, i128 0 acquire
  ret void
}
; CHECK: atomic128_xor_acquire
; CHECK: call i128 @__tsan_atomic128_fetch_xor(i128* %a, i128 0, i32 2)

define void @atomic128_xchg_release(i128* %a) nounwind uwtable {
entry:
  atomicrmw xchg i128* %a, i128 0 release
  ret void
}
; CHECK: atomic128_xchg_release
; CHECK: call i128 @__tsan_atomic128_exchange(i128* %a, i128 0, i32 3)

define void @atomic128_add_release(i128* %a) nounwind uwtable {
entry:
  atomicrmw add i128* %a, i128 0 release
  ret void
}
; CHECK: atomic128_add_release
; CHECK: call i128 @__tsan_atomic128_fetch_add(i128* %a, i128 0, i32 3)

define void @atomic128_sub_release(i128* %a) nounwind uwtable {
entry:
  atomicrmw sub i128* %a, i128 0 release
  ret void
}
; CHECK: atomic128_sub_release
; CHECK: call i128 @__tsan_atomic128_fetch_sub(i128* %a, i128 0, i32 3)

define void @atomic128_and_release(i128* %a) nounwind uwtable {
entry:
  atomicrmw and i128* %a, i128 0 release
  ret void
}
; CHECK: atomic128_and_release
; CHECK: call i128 @__tsan_atomic128_fetch_and(i128* %a, i128 0, i32 3)

define void @atomic128_or_release(i128* %a) nounwind uwtable {
entry:
  atomicrmw or i128* %a, i128 0 release
  ret void
}
; CHECK: atomic128_or_release
; CHECK: call i128 @__tsan_atomic128_fetch_or(i128* %a, i128 0, i32 3)

define void @atomic128_xor_release(i128* %a) nounwind uwtable {
entry:
  atomicrmw xor i128* %a, i128 0 release
  ret void
}
; CHECK: atomic128_xor_release
; CHECK: call i128 @__tsan_atomic128_fetch_xor(i128* %a, i128 0, i32 3)

define void @atomic128_xchg_acq_rel(i128* %a) nounwind uwtable {
entry:
  atomicrmw xchg i128* %a, i128 0 acq_rel
  ret void
}
; CHECK: atomic128_xchg_acq_rel
; CHECK: call i128 @__tsan_atomic128_exchange(i128* %a, i128 0, i32 4)

define void @atomic128_add_acq_rel(i128* %a) nounwind uwtable {
entry:
  atomicrmw add i128* %a, i128 0 acq_rel
  ret void
}
; CHECK: atomic128_add_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_add(i128* %a, i128 0, i32 4)

define void @atomic128_sub_acq_rel(i128* %a) nounwind uwtable {
entry:
  atomicrmw sub i128* %a, i128 0 acq_rel
  ret void
}
; CHECK: atomic128_sub_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_sub(i128* %a, i128 0, i32 4)

define void @atomic128_and_acq_rel(i128* %a) nounwind uwtable {
entry:
  atomicrmw and i128* %a, i128 0 acq_rel
  ret void
}
; CHECK: atomic128_and_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_and(i128* %a, i128 0, i32 4)

define void @atomic128_or_acq_rel(i128* %a) nounwind uwtable {
entry:
  atomicrmw or i128* %a, i128 0 acq_rel
  ret void
}
; CHECK: atomic128_or_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_or(i128* %a, i128 0, i32 4)

define void @atomic128_xor_acq_rel(i128* %a) nounwind uwtable {
entry:
  atomicrmw xor i128* %a, i128 0 acq_rel
  ret void
}
; CHECK: atomic128_xor_acq_rel
; CHECK: call i128 @__tsan_atomic128_fetch_xor(i128* %a, i128 0, i32 4)

define void @atomic128_xchg_seq_cst(i128* %a) nounwind uwtable {
entry:
  atomicrmw xchg i128* %a, i128 0 seq_cst
  ret void
}
; CHECK: atomic128_xchg_seq_cst
; CHECK: call i128 @__tsan_atomic128_exchange(i128* %a, i128 0, i32 5)

define void @atomic128_add_seq_cst(i128* %a) nounwind uwtable {
entry:
  atomicrmw add i128* %a, i128 0 seq_cst
  ret void
}
; CHECK: atomic128_add_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_add(i128* %a, i128 0, i32 5)

define void @atomic128_sub_seq_cst(i128* %a) nounwind uwtable {
entry:
  atomicrmw sub i128* %a, i128 0 seq_cst
  ret void
}
; CHECK: atomic128_sub_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_sub(i128* %a, i128 0, i32 5)

define void @atomic128_and_seq_cst(i128* %a) nounwind uwtable {
entry:
  atomicrmw and i128* %a, i128 0 seq_cst
  ret void
}
; CHECK: atomic128_and_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_and(i128* %a, i128 0, i32 5)

define void @atomic128_or_seq_cst(i128* %a) nounwind uwtable {
entry:
  atomicrmw or i128* %a, i128 0 seq_cst
  ret void
}
; CHECK: atomic128_or_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_or(i128* %a, i128 0, i32 5)

define void @atomic128_xor_seq_cst(i128* %a) nounwind uwtable {
entry:
  atomicrmw xor i128* %a, i128 0 seq_cst
  ret void
}
; CHECK: atomic128_xor_seq_cst
; CHECK: call i128 @__tsan_atomic128_fetch_xor(i128* %a, i128 0, i32 5)

define void @atomic128_cas_monotonic(i128* %a) nounwind uwtable {
entry:
  cmpxchg i128* %a, i128 0, i128 1 monotonic
  ret void
}
; CHECK: atomic128_cas_monotonic
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(i128* %a, i128 0, i128 1, i32 0)

define void @atomic128_cas_acquire(i128* %a) nounwind uwtable {
entry:
  cmpxchg i128* %a, i128 0, i128 1 acquire
  ret void
}
; CHECK: atomic128_cas_acquire
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(i128* %a, i128 0, i128 1, i32 2)

define void @atomic128_cas_release(i128* %a) nounwind uwtable {
entry:
  cmpxchg i128* %a, i128 0, i128 1 release
  ret void
}
; CHECK: atomic128_cas_release
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(i128* %a, i128 0, i128 1, i32 3)

define void @atomic128_cas_acq_rel(i128* %a) nounwind uwtable {
entry:
  cmpxchg i128* %a, i128 0, i128 1 acq_rel
  ret void
}
; CHECK: atomic128_cas_acq_rel
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(i128* %a, i128 0, i128 1, i32 4)

define void @atomic128_cas_seq_cst(i128* %a) nounwind uwtable {
entry:
  cmpxchg i128* %a, i128 0, i128 1 seq_cst
  ret void
}
; CHECK: atomic128_cas_seq_cst
; CHECK: call i128 @__tsan_atomic128_compare_exchange_val(i128* %a, i128 0, i128 1, i32 5)

define void @atomic_signal_fence_acquire() nounwind uwtable {
entry:
  fence singlethread acquire
  ret void
}
; CHECK: atomic_signal_fence_acquire
; CHECK: call void @__tsan_atomic_signal_fence(i32 2)

define void @atomic_thread_fence_acquire() nounwind uwtable {
entry:
  fence  acquire
  ret void
}
; CHECK: atomic_thread_fence_acquire
; CHECK: call void @__tsan_atomic_thread_fence(i32 2)

define void @atomic_signal_fence_release() nounwind uwtable {
entry:
  fence singlethread release
  ret void
}
; CHECK: atomic_signal_fence_release
; CHECK: call void @__tsan_atomic_signal_fence(i32 3)

define void @atomic_thread_fence_release() nounwind uwtable {
entry:
  fence  release
  ret void
}
; CHECK: atomic_thread_fence_release
; CHECK: call void @__tsan_atomic_thread_fence(i32 3)

define void @atomic_signal_fence_acq_rel() nounwind uwtable {
entry:
  fence singlethread acq_rel
  ret void
}
; CHECK: atomic_signal_fence_acq_rel
; CHECK: call void @__tsan_atomic_signal_fence(i32 4)

define void @atomic_thread_fence_acq_rel() nounwind uwtable {
entry:
  fence  acq_rel
  ret void
}
; CHECK: atomic_thread_fence_acq_rel
; CHECK: call void @__tsan_atomic_thread_fence(i32 4)

define void @atomic_signal_fence_seq_cst() nounwind uwtable {
entry:
  fence singlethread seq_cst
  ret void
}
; CHECK: atomic_signal_fence_seq_cst
; CHECK: call void @__tsan_atomic_signal_fence(i32 5)

define void @atomic_thread_fence_seq_cst() nounwind uwtable {
entry:
  fence  seq_cst
  ret void
}
; CHECK: atomic_thread_fence_seq_cst
; CHECK: call void @__tsan_atomic_thread_fence(i32 5)
