; RUN: opt < %s -aa-pipeline=basic-aa,scoped-noalias-aa -passes='print<memoryssa>' -disable-output 2>&1 | FileCheck %s

; Test that ScopedNoAliasAA::getModRefInfo(FenceInst) uses scoped noalias
; metadata to prove a fence cannot affect a given memory location.
; MemorySSA exposes this: when the fence is NoModRef w.r.t. a load, the
; load's clobbering access is liveOnEntry (not the fence).

define i32 @fence_noalias(ptr %p) {
; CHECK-LABEL: MemorySSA for function: fence_noalias
; Fence has !noalias covering the load's scope -> not a clobber.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v1 = load
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v2 = load
  %v1 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !3
  fence syncscope("workgroup") release, !noalias !5
  %v2 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !3
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

define i32 @fence_alias_scope(ptr %p) {
; CHECK-LABEL: MemorySSA for function: fence_alias_scope
; Symmetric: fence has !alias.scope, load has !noalias covering it.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v1 = load
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v2 = load
  %v1 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !7
  fence syncscope("workgroup") release, !alias.scope !7
  %v2 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !7
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

define i32 @fence_no_metadata(ptr %p) {
; CHECK-LABEL: MemorySSA for function: fence_no_metadata
; No metadata on fence -> fence is a clobber for the second load.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v1 = load
; CHECK: MemoryUse([[FENCE:.*]])
; CHECK-NEXT: %v2 = load
  %v1 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !3
  fence syncscope("workgroup") release
  %v2 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !3
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

define i32 @fence_partial_noalias(ptr %p) {
; CHECK-LABEL: MemorySSA for function: fence_partial_noalias
; Fence has !noalias for other_arg_scope only, load is in arg_scope.
; Scopes don't match -> fence is still a clobber.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v1 = load
; CHECK: MemoryUse([[FENCE2:.*]])
; CHECK-NEXT: %v2 = load
  %v1 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !3
  fence syncscope("workgroup") release, !noalias !3
  %v2 = load i32, ptr %p, align 4, !alias.scope !0, !noalias !3
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

!0 = !{!1}
!1 = distinct !{!1, !2, !"arg_scope"}
!2 = distinct !{!2, !"kernel_domain"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"other_arg_scope"}
!5 = !{!1, !4}
!6 = distinct !{!6, !2, !"fence_sync_scope"}
!7 = !{!6}
