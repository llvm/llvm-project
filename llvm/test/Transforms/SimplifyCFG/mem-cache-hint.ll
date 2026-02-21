; RUN: opt -passes=simplifycfg -S < %s | FileCheck %s

; Test that SimplifyCFG speculates the conditional load and that
; dropUBImplyingAttrsAndMetadata() keeps !mem.cache_hint on the
; speculated load. mem.cache_hint is a performance hint and does not
; imply UB, so it is safe to preserve.

; CHECK-LABEL: @speculate_keeps_hint
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[V:%.*]] = load i32, ptr %p{{.*}} !mem.cache_hint
; CHECK-NEXT:    [[SEL:%.*]] = select i1 %c, i32 [[V]], i32 0
; CHECK-NEXT:    ret i32 [[SEL]]
define i32 @speculate_keeps_hint(i1 %c, ptr dereferenceable(4) align 4 %p) {
entry:
  br i1 %c, label %if, label %join
if:
  %v = load i32, ptr %p, !mem.cache_hint !0
  br label %join
join:
  %phi = phi i32 [ %v, %if ], [ 0, %entry ]
  ret i32 %phi
}

!0 = !{ i32 0, !1 }
!1 = !{ !"nvvm.l1_eviction", !"first" }
