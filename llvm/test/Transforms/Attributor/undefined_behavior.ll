; RUN: opt --attributor --attributor-disable=false -S < %s | FileCheck %s --check-prefix=ATTRIBUTOR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for the "undefined behavior" abstract function attribute.
; We want to verify that whenever undefined behavior is assumed, the code becomes unreachable.
; We use FIXME's to indicate problems and missing attributes.

; -- Load tests --

define void @load_wholly_unreachable() {
; ATTRIBUTOR-LABEL: @load_wholly_unreachable(
; ATTRIBUTOR-NEXT:    unreachable
;
  %a = load i32, i32* null
  ret void
}

define void @loads_wholly_unreachable() {
; ATTRIBUTOR-LABEL: @loads_wholly_unreachable(
; ATTRIBUTOR-NEXT:    unreachable
;
  %a = load i32, i32* null
  %b = load i32, i32* null
  ret void
}


define void @load_single_bb_unreachable(i1 %cond) {
; ATTRIBUTOR-LABEL: @load_single_bb_unreachable(
; ATTRIBUTOR-NEXT:    br i1 [[COND:%.*]], label [[T:%.*]], label [[E:%.*]]
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    ret void
;
  br i1 %cond, label %t, label %e
t:
  %b = load i32, i32* null
  br label %e
e:
  ret void
}

define void @load_null_pointer_is_defined() "null-pointer-is-valid"="true" {
; ATTRIBUTOR-LABEL: @load_null_pointer_is_defined(
; ATTRIBUTOR-NEXT:    ret void
;
  %a = load i32, i32* null
  ret void
}

define internal i32* @ret_null() {
  ret i32* null
}

; FIXME: null is propagated but the instruction
; is not changed to unreachable.
define i32 @load_null_propagated() {
; ATTRIBUTOR-LABEL: @load_null_propagated(
; ATTRIBUTOR-NEXT:    [[A:%.*]] = load i32, i32* null
; ATTRIBUTOR-NEXT:    ret i32 [[A]]
;
  %ptr = call i32* @ret_null()
  %a = load i32, i32* %ptr
  ret i32 %a
}

; -- Store tests --

define void @store_wholly_unreachable() {
; ATTRIBUTOR-LABEL: @store_wholly_unreachable(
; ATTRIBUTOR-NEXT:    unreachable
;
  store i32 5, i32* null
  ret void
}

define void @store_single_bb_unreachable(i1 %cond) {
; ATTRIBUTOR-LABEL: @store_single_bb_unreachable(
; ATTRIBUTOR-NEXT:    br i1 [[COND:%.*]], label [[T:%.*]], label [[E:%.*]]
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    ret void
;
  br i1 %cond, label %t, label %e
t:
  store i32 5, i32* null
  br label %e
e:
  ret void
}

define void @store_null_pointer_is_defined() "null-pointer-is-valid"="true" {
; ATTRIBUTOR-LABEL: @store_null_pointer_is_defined(
; ATTRIBUTOR-NEXT:    store i32 5, i32* null
; ATTRIBUTOR-NEXT:    ret void
;
  store i32 5, i32* null
  ret void
}

; -- AtomicRMW tests --

define void @atomicrmw_wholly_unreachable() {
; ATTRIBUTOR-LABEL: @atomicrmw_wholly_unreachable(
; ATTRIBUTOR-NEXT:    unreachable
;
  %a = atomicrmw add i32* null, i32 1 acquire
  ret void
}

define void @atomicrmw_single_bb_unreachable(i1 %cond) {
; ATTRIBUTOR-LABEL: @atomicrmw_single_bb_unreachable(
; ATTRIBUTOR-NEXT:    br i1 [[COND:%.*]], label [[T:%.*]], label [[E:%.*]]
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    ret void
;
  br i1 %cond, label %t, label %e
t:
  %a = atomicrmw add i32* null, i32 1 acquire
  br label %e
e:
  ret void
}

define void @atomicrmw_null_pointer_is_defined() "null-pointer-is-valid"="true" {
; ATTRIBUTOR-LABEL: @atomicrmw_null_pointer_is_defined(
; ATTRIBUTOR-NEXT:    [[A:%.*]] = atomicrmw add i32* null, i32 1 acquire
; ATTRIBUTOR-NEXT:    ret void
;
  %a = atomicrmw add i32* null, i32 1 acquire
  ret void
}

; -- AtomicCmpXchg tests --

define void @atomiccmpxchg_wholly_unreachable() {
; ATTRIBUTOR-LABEL: @atomiccmpxchg_wholly_unreachable(
; ATTRIBUTOR-NEXT:    unreachable
;
  %a = cmpxchg i32* null, i32 2, i32 3 acq_rel monotonic
  ret void
}

define void @atomiccmpxchg_single_bb_unreachable(i1 %cond) {
; ATTRIBUTOR-LABEL: @atomiccmpxchg_single_bb_unreachable(
; ATTRIBUTOR-NEXT:    br i1 [[COND:%.*]], label [[T:%.*]], label [[E:%.*]]
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    ret void
;
  br i1 %cond, label %t, label %e
t:
  %a = cmpxchg i32* null, i32 2, i32 3 acq_rel monotonic
  br label %e
e:
  ret void
}

define void @atomiccmpxchg_null_pointer_is_defined() "null-pointer-is-valid"="true" {
; ATTRIBUTOR-LABEL: @atomiccmpxchg_null_pointer_is_defined(
; ATTRIBUTOR-NEXT:    [[A:%.*]] = cmpxchg i32* null, i32 2, i32 3 acq_rel monotonic
; ATTRIBUTOR-NEXT:    ret void
;
  %a = cmpxchg i32* null, i32 2, i32 3 acq_rel monotonic
  ret void
}

; Note: The unreachable on %t and %e is _not_ from AAUndefinedBehavior

define i32 @cond_br_on_undef() {
; ATTRIBUTOR-LABEL: @cond_br_on_undef(
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    unreachable
;

  br i1 undef, label %t, label %e
t:
  ret i32 1
e:
  ret i32 2
}

; More complicated branching
define void @cond_br_on_undef2(i1 %cond) {
; ATTRIBUTOR-LABEL: @cond_br_on_undef2(
; ATTRIBUTOR-NEXT:    br i1 [[COND:%.*]], label [[T1:%.*]], label [[E1:%.*]]
; ATTRIBUTOR:       t1:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       t2:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e2:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e1:
; ATTRIBUTOR-NEXT:    ret void
;

  ; Valid branch - verify that this is not converted
  ; to unreachable.
  br i1 %cond, label %t1, label %e1
t1:
  br i1 undef, label %t2, label %e2
t2:
  ret void
e2:
  ret void
e1:
  ret void
}

define i1 @ret_undef() {
  ret i1 undef
}

define void @cond_br_on_undef_interproc() {
; ATTRIBUTOR-LABEL: @cond_br_on_undef_interproc(
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    unreachable
  
  %cond = call i1 @ret_undef()
  br i1 %cond, label %t, label %e
t:
  ret void
e:
  ret void
}

define i1 @ret_undef2() {
  br i1 true, label %t, label %e
t:
  ret i1 undef
e:
  ret i1 undef
}

; More complicated interproc deduction of undef
define void @cond_br_on_undef_interproc2() {
; ATTRIBUTOR-LABEL: @cond_br_on_undef_interproc2(
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    unreachable
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    unreachable
  %cond = call i1 @ret_undef2()
  br i1 %cond, label %t, label %e
t:
  ret void
e:
  ret void
}

; Branch on undef that depends on propagation of
; undef of a previous instruction.
define i32 @cond_br_on_undef3() {
; ATTRIBUTOR-LABEL: @cond_br_on_undef3(
; ATTRIBUTOR-NEXT:    br label %t
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    ret i32 1
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    unreachable

  %cond = icmp ne i32 1, undef
  br i1 %cond, label %t, label %e
t:
  ret i32 1
e:
  ret i32 2
}

; Branch on undef because of uninitialized value.
; FIXME: Currently it doesn't propagate the undef.
define i32 @cond_br_on_undef_uninit() {
; ATTRIBUTOR-LABEL: @cond_br_on_undef_uninit(
; ATTRIBUTOR-NEXT:    %alloc = alloca i1
; ATTRIBUTOR-NEXT:    %cond = load i1, i1* %alloc
; ATTRIBUTOR-NEXT:    br i1 %cond, label %t, label %e
; ATTRIBUTOR:       t:
; ATTRIBUTOR-NEXT:    ret i32 1
; ATTRIBUTOR:       e:
; ATTRIBUTOR-NEXT:    ret i32 2
  
  %alloc = alloca i1
  %cond = load i1, i1* %alloc
  br i1 %cond, label %t, label %e
t:
  ret i32 1
e:
  ret i32 2
}

; Note that the `load` has UB (so it will be changed to unreachable)
; and the branch is a terminator that can be constant-folded.
; We want to test that doing both won't cause a segfault.
define internal i32 @callee(i1 %C, i32* %A) {
; ATTRIBUTOR-NOT: @callee(
;
entry:
  %A.0 = load i32, i32* null
  br i1 %C, label %T, label %F

T:
  ret i32 %A.0

F:
  ret i32 1
}

define i32 @foo() {
; ATTRIBUTOR-LABEL: @foo()
; ATTRIBUTOR-NEXT:    ret i32 1
  %X = call i32 @callee(i1 false, i32* null)
  ret i32 %X
}
