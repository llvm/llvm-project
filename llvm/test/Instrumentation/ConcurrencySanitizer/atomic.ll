; RUN: opt < %s -passes='function(csan)' -S -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
; RUN: opt < %s -passes='function(csan)' -S -mtriple=amdgcn-amd-amdhsa | FileCheck %s

define i32 @atomic_load(ptr %a) sanitize_concurrency {
entry:
  %v = load atomic i32, ptr %a seq_cst, align 4
  ret i32 %v
}
; CHECK-LABEL: @atomic_load(
; CHECK: call void @__csan_read4(ptr %a, i32 1)
; CHECK-NEXT: %v = load atomic i32, ptr %a seq_cst, align 4

define void @atomic_store(ptr %a, i32 %v) sanitize_concurrency {
entry:
  store atomic i32 %v, ptr %a release, align 4
  ret void
}
; CHECK-LABEL: @atomic_store(
; CHECK: call void @__csan_write4(ptr %a, i32 1)
; CHECK-NEXT: store atomic i32 %v, ptr %a release, align 4

define i32 @atomic_rmw(ptr %a, i32 %v) sanitize_concurrency {
entry:
  %old = atomicrmw add ptr %a, i32 %v seq_cst
  ret i32 %old
}
; CHECK-LABEL: @atomic_rmw(
; CHECK: call void @__csan_read_write4(ptr %a, i32 3)
; CHECK-NEXT: %old = atomicrmw add ptr %a, i32 %v seq_cst

define i32 @atomic_cas(ptr %a, i32 %cmp, i32 %new) sanitize_concurrency {
entry:
  %pair = cmpxchg ptr %a, i32 %cmp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}
; CHECK-LABEL: @atomic_cas(
; CHECK: call void @__csan_read_write4(ptr %a, i32 3)
; CHECK-NEXT: %pair = cmpxchg ptr %a, i32 %cmp, i32 %new seq_cst seq_cst

define void @atomic_fence() sanitize_concurrency {
entry:
  fence seq_cst
  ret void
}
; CHECK-LABEL: @atomic_fence(
; CHECK: call void @__csan_atomic_thread_fence(i32 5)
; CHECK-NEXT: fence seq_cst

define void @atomic_signal_fence() sanitize_concurrency {
entry:
  fence syncscope("singlethread") seq_cst
  ret void
}
; CHECK-LABEL: @atomic_signal_fence(
; CHECK: call void @__csan_atomic_signal_fence(i32 5)
; CHECK-NEXT: fence syncscope("singlethread") seq_cst

define void @atomic_fence_orderings() sanitize_concurrency {
entry:
  fence acquire
  fence release
  fence acq_rel
  ret void
}
; CHECK-LABEL: @atomic_fence_orderings(
; CHECK: call void @__csan_atomic_thread_fence(i32 2)
; CHECK-NEXT: fence acquire
; CHECK: call void @__csan_atomic_thread_fence(i32 3)
; CHECK-NEXT: fence release
; CHECK: call void @__csan_atomic_thread_fence(i32 4)
; CHECK-NEXT: fence acq_rel

define i32 @suppressed(ptr %a) {
entry:
  %v = load atomic i32, ptr %a seq_cst, align 4
  ret i32 %v
}
; CHECK-LABEL: @suppressed(
; CHECK-NEXT: entry:
; CHECK-NEXT: %v = load atomic i32, ptr %a seq_cst, align 4
; CHECK-NEXT: ret i32 %v

; CHECK-NOT: @__tsan
; CHECK-NOT: @__csan_atomic8
; CHECK-NOT: @__csan_atomic32
