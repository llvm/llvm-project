; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -icp-max-prom=1 -S | \
; RUN:   FileCheck %s --check-prefix=LIMIT
; RUN: opt < %s -passes='cgscc(inline)' \
; RUN:   -icp-max-static-target-traversal=2 -S | \
; RUN:   FileCheck %s --check-prefix=TRAVERSAL
; RUN: opt < %s -passes='cgscc(inline<only-mandatory>)' -S | \
; RUN:   FileCheck %s --check-prefix=MANDATORY
; RUN: opt < %s -passes='cgscc(inline)' -S | \
; RUN:   FileCheck %s --check-prefix=DYNAMIC
; RUN: opt < %s -passes='cgscc(inline),simplifycfg' -S | \
; RUN:   FileCheck %s --check-prefix=ORDER

@fnptr = global ptr @foo
@foo_alias = alias i32 (), ptr @foo
@foo_ifunc = ifunc i32 (), ptr @resolve_foo

define i32 @foo() {
  ret i32 1
}

define i32 @bar() {
  ret i32 2
}

define ptr @resolve_foo() {
  ret ptr @foo
}

declare ptr @get_fnptr()

define i32 @always_foo() alwaysinline {
  ret i32 1
}

define i32 @always_bar() alwaysinline {
  ret i32 2
}

define i32 @musttail_foo(i1 %c) {
  ret i32 1
}

define i32 @musttail_bar(i1 %c) {
  ret i32 2
}

define i32 @addrspace_function() addrspace(1) {
  ret i32 3
}

define i32 @baz() {
  ret i32 3
}

define i32 @qux() {
  ret i32 4
}

define i32 @quux() {
  ret i32 5
}

define i32 @select_callee(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @select_callee(
; CHECK-NOT: call
; CHECK: select i1 %c, i32 1, i32 2
; CHECK: ret i32
; LIMIT-LABEL: define i32 @select_callee(
; LIMIT: %callee = select i1 %c, ptr @foo, ptr @bar
; LIMIT: call i32 %callee()
; TRAVERSAL-LABEL: define i32 @select_callee(
; TRAVERSAL: %callee = select i1 %c, ptr @foo, ptr @bar
; TRAVERSAL: call i32 %callee()
  %callee = select i1 %c, ptr @foo, ptr @bar
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @mandatory_select_callee(i1 %c) {
; MANDATORY-LABEL: define i32 @mandatory_select_callee(
; MANDATORY-NOT: call i32
; MANDATORY: phi i32 [ 2, %if.false.orig_indirect ], [ 1, %if.true.direct_targ ]
; MANDATORY: ret i32
  %callee = select i1 %c, ptr @always_foo, ptr @always_bar
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @adjacent_indirect_calls(i1 %c1, i1 %c2) {
; CHECK-LABEL: define {{.*}}i32 @adjacent_indirect_calls(
; CHECK-NOT: call
; CHECK: %[[FIRST:.*]] = select i1 %c1, i32 1, i32 2
; CHECK: %[[SECOND:.*]] = select i1 %c2, i32 1, i32 2
; CHECK: %sum = add {{.*}}i32
  %callee1 = select i1 %c1, ptr @foo, ptr @bar
  %result1 = call i32 %callee1()
  %callee2 = select i1 %c2, ptr @foo, ptr @bar
  %result2 = call i32 %callee2()
  %sum = add i32 %result1, %result2
  ret i32 %sum
}

declare i32 @__gxx_personality_v0(...)

define i32 @invoke_callee(i1 %c) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define {{.*}}i32 @invoke_callee(
; CHECK-NOT: call
; CHECK-NOT: invoke
; CHECK: select i1 %c, i32 1, i32 2
; CHECK: ret i32
entry:
  %callee = select i1 %c, ptr @foo, ptr @bar
  %result = invoke i32 %callee()
      to label %normal unwind label %lpad

normal:
  ret i32 %result

lpad:
  %lp = landingpad { ptr, i32 }
      cleanup
  resume { ptr, i32 } %lp
}

define i32 @musttail_callee(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @musttail_callee(
; CHECK-NOT: call
; CHECK: select i1 %c, i32 1, i32 2
; CHECK: ret i32
  %callee = select i1 %c, ptr @musttail_foo, ptr @musttail_bar
  %result = musttail call i32 %callee(i1 %c)
  ret i32 %result
}

define i32 @recursive_leaf(i1 %c) {
  ret i32 1
}

define i32 @recursive_target(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @recursive_target(
; CHECK: %callee = select i1 %c, ptr @recursive_target, ptr @recursive_leaf
; CHECK: call i32 %callee(i1 false)
  %callee = select i1 %c, ptr @recursive_target, ptr @recursive_leaf
  %result = call i32 %callee(i1 false)
  ret i32 %result
}

define i32 @scc_target() {
  %result = call i32 @scc_caller(i1 false)
  ret i32 %result
}

define i32 @scc_caller(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @scc_caller(
; CHECK: ret i32
  %callee = select i1 %c, ptr @scc_target, ptr @foo
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @cyclic_argument_target(i1 %choose, i1 %again, ptr %unknown) {
; DYNAMIC-LABEL: define i32 @cyclic_argument_target(
; DYNAMIC: %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
; DYNAMIC: call i32 %callee()
entry:
  br label %loop

loop:
  %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
  %next = select i1 %choose, ptr %callee, ptr %unknown
  %result = call i32 %callee()
  br i1 %again, label %loop, label %exit

exit:
  ret i32 %result
}

define i32 @cyclic_load_target(i1 %choose, i1 %again) {
; DYNAMIC-LABEL: define i32 @cyclic_load_target(
; DYNAMIC: %callee = phi ptr [ @bar, %entry ], [ %next, %loop ]
; DYNAMIC: call i32 %callee()
entry:
  %unknown = load ptr, ptr @fnptr
  br label %loop

loop:
  %callee = phi ptr [ @bar, %entry ], [ %next, %loop ]
  %next = select i1 %choose, ptr %callee, ptr %unknown
  %result = call i32 %callee()
  br i1 %again, label %loop, label %exit

exit:
  ret i32 %result
}

define i32 @cyclic_call_target(i1 %choose, i1 %again) {
; DYNAMIC-LABEL: define i32 @cyclic_call_target(
; DYNAMIC: %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
; DYNAMIC: call i32 %callee()
entry:
  %unknown = call ptr @get_fnptr()
  br label %loop

loop:
  %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
  %next = select i1 %choose, ptr %callee, ptr %unknown
  %result = call i32 %callee()
  br i1 %again, label %loop, label %exit

exit:
  ret i32 %result
}

define i32 @cyclic_second_argument_target(i1 %choose, i1 %again, ptr %unknown) {
; DYNAMIC-LABEL: define i32 @cyclic_second_argument_target(
; DYNAMIC: %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
; DYNAMIC: call i32 %callee()
entry:
  br label %loop

loop:
  %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
  %next = select i1 %choose, ptr %callee, ptr %unknown
  %result = call i32 %callee()
  br i1 %again, label %loop, label %exit

exit:
  ret i32 %result
}

define i32 @cyclic_third_argument_target(i1 %choose, i1 %again, ptr %unknown) {
; DYNAMIC-LABEL: define i32 @cyclic_third_argument_target(
; DYNAMIC: %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
; DYNAMIC: call i32 %callee()
entry:
  br label %loop

loop:
  %callee = phi ptr [ @foo, %entry ], [ %next, %loop ]
  %next = select i1 %choose, ptr %callee, ptr %unknown
  %result = call i32 %callee()
  br i1 %again, label %loop, label %exit

exit:
  ret i32 %result
}

define i32 @dynamic_leaf_targets(i1 %c, ptr %argument, ptr %other,
                                 ptr %third) {
; DYNAMIC-LABEL: define i32 @dynamic_leaf_targets(
; DYNAMIC: call i32 %load.target()
; DYNAMIC: call i32 %atomic.target()
; DYNAMIC: call i32 %volatile.target()
; DYNAMIC: call i32 %argument.target()
; DYNAMIC: call i32 %call.target()
; DYNAMIC: call i32 %other.target()
; DYNAMIC: call i32 %third.target()
; DYNAMIC: call i32 %alias.target()
; DYNAMIC: call i32 %ifunc.target()
  %loaded = load ptr, ptr @fnptr
  %atomic = load atomic ptr, ptr @fnptr monotonic, align 8
  %volatile = load volatile ptr, ptr @fnptr
  %returned = call ptr @get_fnptr()
  %load.target = select i1 %c, ptr @foo, ptr %loaded
  %atomic.target = select i1 %c, ptr @foo, ptr %atomic
  %volatile.target = select i1 %c, ptr @foo, ptr %volatile
  %argument.target = select i1 %c, ptr @foo, ptr %argument
  %call.target = select i1 %c, ptr @foo, ptr %returned
  %other.target = select i1 %c, ptr @foo, ptr %other
  %third.target = select i1 %c, ptr @foo, ptr %third
  %alias.target = select i1 %c, ptr @foo, ptr @foo_alias
  %ifunc.target = select i1 %c, ptr @foo, ptr @foo_ifunc
  %r0 = call i32 %load.target()
  %r1 = call i32 %atomic.target()
  %r2 = call i32 %volatile.target()
  %r3 = call i32 %argument.target()
  %r4 = call i32 %call.target()
  %r5 = call i32 %other.target()
  %r6 = call i32 %third.target()
  %r7 = call i32 %alias.target()
  %r8 = call i32 %ifunc.target()
  %s0 = add i32 %r0, %r1
  %s1 = add i32 %r2, %r3
  %s2 = add i32 %r4, %r5
  %s3 = add i32 %r6, %r7
  %s4 = add i32 %s0, %s1
  %s5 = add i32 %s2, %s3
  %s6 = add i32 %s4, %s5
  %s7 = add i32 %s6, %r8
  ret i32 %s7
}

define i32 @different_function_pointer_representation(i1 %c) {
; DYNAMIC-LABEL: define i32 @different_function_pointer_representation(
; DYNAMIC: %callee = select i1 %c, ptr @foo, ptr addrspacecast (ptr addrspace(1) @addrspace_function to ptr)
; DYNAMIC: call i32 %callee()
  %callee = select i1 %c, ptr @foo, ptr addrspacecast (ptr addrspace(1) @addrspace_function to ptr)
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @duplicate_target(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @duplicate_target(
; CHECK-NOT: call
; CHECK: ret i32 1
; DYNAMIC-LABEL: define i32 @duplicate_target(
; DYNAMIC-NOT: call i32
; DYNAMIC: ret i32 1
  %callee = select i1 %c, ptr @foo, ptr @foo
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @phi_callee(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @phi_callee(
; CHECK-NOT: call
; CHECK: select i1 %c, i32 1, i32 2
; CHECK: ret i32
entry:
  br i1 %c, label %left, label %right

left:
  br label %join

right:
  br label %join

join:
  %callee = phi ptr [ @foo, %left ], [ @bar, %right ]
  %result = call i32 %callee()
  ret i32 %result
}

define ptr @choose_callee(i1 %c) {
  %callee = select i1 %c, ptr @foo, ptr @bar
  ret ptr %callee
}

define i32 @exposed_after_inlining(i1 %c) {
; ORDER-LABEL: define i32 @exposed_after_inlining(
; ORDER: %callee.i = select i1 %c, ptr @foo, ptr @bar
; ORDER: icmp eq ptr %callee.i, @foo
; ORDER-NOT: icmp eq ptr %callee.i, @bar
; ORDER-NOT: call i32 %callee.i
; ORDER: ret i32
  %callee = call ptr @choose_callee(i1 %c)
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @unknown_target(i1 %c, ptr %unknown) {
; CHECK-LABEL: define {{.*}}i32 @unknown_target(
; CHECK: %callee = select i1 %c, ptr @foo, ptr %unknown
; CHECK: call i32 %callee()
  %callee = select i1 %c, ptr @foo, ptr %unknown
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @noinline_foo() noinline {
  ret i32 1
}

define i32 @noinline_bar() noinline {
  ret i32 2
}

define i32 @unprofitable(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @unprofitable(
; CHECK: %callee = select i1 %c, ptr @noinline_foo, ptr @noinline_bar
; CHECK: call i32 %callee()
  %callee = select i1 %c, ptr @noinline_foo, ptr @noinline_bar
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @partially_profitable(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @partially_profitable(
; CHECK: %callee = select i1 %c, ptr @foo, ptr @noinline_bar
; CHECK: call i32 %callee()
  %callee = select i1 %c, ptr @foo, ptr @noinline_bar
  %result = call i32 %callee()
  ret i32 %result
}

; Inline cost uses a strict comparison against the threshold. Verify that
; static promotion follows the same boundary as an ordinary direct call.
define i32 @just_below_threshold(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @just_below_threshold(
; CHECK-NOT: call
; CHECK: select i1 %c, i32 1, i32 2
; CHECK: ret i32
  %callee = select i1 %c, ptr @foo, ptr @bar
  %result = call i32 %callee() "function-inline-cost"="122" "function-inline-threshold"="123"
  ret i32 %result
}

define i32 @at_threshold(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @at_threshold(
; CHECK: %callee = select i1 %c, ptr @foo, ptr @bar
; CHECK: call i32 %callee()
  %callee = select i1 %c, ptr @foo, ptr @bar
  %result = call i32 %callee() "function-inline-cost"="123" "function-inline-threshold"="123"
  ret i32 %result
}

define i32 @three_candidates(i2 %selector) {
; CHECK-LABEL: define {{.*}}i32 @three_candidates(
; CHECK-NOT: call
; CHECK: ret i32
  %is0 = icmp eq i2 %selector, 0
  %is1 = icmp eq i2 %selector, 1
  %middle = select i1 %is1, ptr @bar, ptr @baz
  %callee = select i1 %is0, ptr @foo, ptr %middle
  %result = call i32 %callee()
  ret i32 %result
}

define i32 @four_candidates(i3 %selector) {
; CHECK-LABEL: define {{.*}}i32 @four_candidates(
; CHECK: %callee = select i1 %is0, ptr @foo, ptr %middle
; CHECK: call i32 %callee()
  %is0 = icmp eq i3 %selector, 0
  %is1 = icmp eq i3 %selector, 1
  %is2 = icmp eq i3 %selector, 2
  %lower = select i1 %is2, ptr @baz, ptr @qux
  %middle = select i1 %is1, ptr @bar, ptr %lower
  %callee = select i1 %is0, ptr @foo, ptr %middle
  %result = call i32 %callee()
  ret i32 %result
}

declare i32 @declaration()

define i32 @unavailable_definition(i1 %c) {
; CHECK-LABEL: define {{.*}}i32 @unavailable_definition(
; CHECK: %callee = select i1 %c, ptr @foo, ptr @declaration
; CHECK: call i32 %callee()
  %callee = select i1 %c, ptr @foo, ptr @declaration
  %result = call i32 %callee()
  ret i32 %result
}

define i64 @different_signature(i64 %x) {
  ret i64 %x
}

define i32 @incompatible_signature(i1 %c, i32 %x) {
; CHECK-LABEL: define {{.*}}i32 @incompatible_signature(
; CHECK: %callee = select i1 %c, ptr @foo, ptr @different_signature
; CHECK: call i32 %callee(i32 %x)
  %callee = select i1 %c, ptr @foo, ptr @different_signature
  %result = call i32 %callee(i32 %x)
  ret i32 %result
}
