; Check that a musttail call is not inlined when the callee itself ends in an
; indirect musttail call, i.e. when inlining would only extend a tail call
; dispatch chain, as found in threaded interpreters.

; RUN: opt < %s -passes=inline -S | FileCheck %s

declare ptr @get_handler(ptr)

; The dispatch site has been promoted to a direct musttail call to @handler_b
; guarded by a compare, with the original indirect call as the fallback.

define i64 @handler_a(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @handler_a(
; CHECK-NOT: musttail call i64 @handler_c
; CHECK: musttail call i64 @handler_b(
entry:
  %next = call ptr @get_handler(ptr %pc)
  %cmp = icmp eq ptr %next, @handler_b
  br i1 %cmp, label %direct, label %indirect

direct:
  %ret0 = musttail call i64 @handler_b(ptr %pc, ptr %regs)
  ret i64 %ret0

indirect:
  %ret1 = musttail call i64 %next(ptr %pc, ptr %regs)
  ret i64 %ret1
}

define i64 @handler_b(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @handler_b(
entry:
  %next = call ptr @get_handler(ptr %pc)
  %cmp = icmp eq ptr %next, @handler_c
  br i1 %cmp, label %direct, label %indirect

direct:
  %ret0 = musttail call i64 @handler_c(ptr %pc, ptr %regs)
  ret i64 %ret0

indirect:
  %ret1 = musttail call i64 %next(ptr %pc, ptr %regs)
  ret i64 %ret1
}

define i64 @handler_c(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @handler_c(
entry:
  %next = call ptr @get_handler(ptr %pc)
  %ret = musttail call i64 %next(ptr %pc, ptr %regs)
  ret i64 %ret
}

; A callee that ends in an indirect musttail call is still inlined into a call
; site that is not itself a musttail call: no dispatch chain is extended there.

define i64 @run(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @run(
; CHECK-NOT: call i64 @handler_c(
entry:
  %ret = call i64 @handler_c(ptr %pc, ptr %regs)
  ret i64 %ret
}

; A musttail callee whose own tail call is direct is a bounded chain and is
; still inlined.

define i64 @tail_to_direct(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @tail_to_direct(
; CHECK-NOT: musttail call i64 @direct_leaf(
; CHECK: musttail call i64 @leaf(
entry:
  %ret = musttail call i64 @direct_leaf(ptr %pc, ptr %regs)
  ret i64 %ret
}

define i64 @direct_leaf(ptr %pc, ptr %regs) {
entry:
  %ret = musttail call i64 @leaf(ptr %pc, ptr %regs)
  ret i64 %ret
}

declare i64 @leaf(ptr, ptr)

; Inlining the sole call to a local function moves its body rather than
; duplicating it, so the dispatch chain is not extended and inlining happens.

define i64 @sole_caller(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @sole_caller(
; CHECK-NOT: musttail call i64 @local_dispatch(
; CHECK: musttail call i64 %{{.*}}(ptr %pc, ptr %regs)
entry:
  %ret = musttail call i64 @local_dispatch(ptr %pc, ptr %regs)
  ret i64 %ret
}

define internal i64 @local_dispatch(ptr %pc, ptr %regs) {
entry:
  %next = call ptr @get_handler(ptr %pc)
  %ret = musttail call i64 %next(ptr %pc, ptr %regs)
  ret i64 %ret
}
