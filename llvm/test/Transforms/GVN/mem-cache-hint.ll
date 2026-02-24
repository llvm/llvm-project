; RUN: opt -passes=gvn -S < %s | FileCheck %s

; Test that combineMetadata() preserves !mem.cache_hint only when present on
; both instructions being merged.

; Both loads have !mem.cache_hint → preserved after GVN deduplication.
; CHECK-LABEL: @both_hint
; CHECK: load i64, ptr %p{{.*}} !mem.cache_hint
define i64 @both_hint(ptr %p) {
  %a = load i64, ptr %p, !mem.cache_hint !0
  %b = load i64, ptr %p, !mem.cache_hint !0
  %c = add i64 %a, %b
  ret i64 %c
}

; Only one load has !mem.cache_hint → dropped after GVN deduplication.
; CHECK-LABEL: @one_hint
; CHECK: load
; CHECK-NOT: !mem.cache_hint
define i64 @one_hint(ptr %p) {
  %a = load i64, ptr %p
  %b = load i64, ptr %p, !mem.cache_hint !0
  %c = add i64 %a, %b
  ret i64 %c
}

; Both loads have !mem.cache_hint but with different payloads
; The merged result is currently undefined. 
; TODO: delegate to TTI to let targets decide how to merge differing payloads.
; CHECK-LABEL: @diff_hint
; CHECK: load i64, ptr %p{{.*}} !mem.cache_hint
define i64 @diff_hint(ptr %p) {
  %a = load i64, ptr %p, !mem.cache_hint !0
  %b = load i64, ptr %p, !mem.cache_hint !2
  %c = add i64 %a, %b
  ret i64 %c
}

!0 = !{ i32 0, !1 }
!1 = !{ !"nvvm.l1_eviction", !"first" }
!2 = !{ i32 0, !3 }
!3 = !{ !"nvvm.l1_eviction", !"last" }
