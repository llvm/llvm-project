; RUN: opt --attributor --attributor-disable=false -S < %s | FileCheck %s --check-prefix=ATTRIBUTOR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for the "undefined behavior" abstract function attribute.
; We want to verify that whenever undefined behavior is assumed, the code becomes unreachable.
; We use FIXME's to indicate problems and missing attributes.

; -- Load tests --

; ATTRIBUTOR-LABEL: define void @load_wholly_unreachable()
define void @load_wholly_unreachable() {
; ATTRIBUTOR-NEXT: unreachable
  %a = load i32, i32* null
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
; ATTRIBUTOR-NEXT:    [[A:%.*]] = load i32, i32* null
; ATTRIBUTOR-NEXT:    ret void
;
  %a = load i32, i32* null
  ret void
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
