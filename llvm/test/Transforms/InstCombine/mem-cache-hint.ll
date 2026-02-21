; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; Test that copyMetadataForLoad() preserves !mem.cache_hint when a load is
; transformed (e.g., load+bitcast folded into a single load).

; CHECK-LABEL: @cast_load_preserve_hint
; CHECK: load i32, ptr %p{{.*}} !mem.cache_hint
define i32 @cast_load_preserve_hint(ptr %p) {
  %l = load float, ptr %p, !mem.cache_hint !0
  %c = bitcast float %l to i32
  ret i32 %c
}

!0 = !{ i32 0, !1 }
!1 = !{ !"nvvm.l1_eviction", !"first" }
