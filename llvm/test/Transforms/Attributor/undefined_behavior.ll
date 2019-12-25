; RUN: opt --attributor --attributor-disable=false -S < %s | FileCheck %s --check-prefix=ATTRIBUTOR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for the "undefined behavior" abstract function attribute.
; We want to verify that whenever undefined behavior is assumed, the code becomes unreachable.
; We use FIXME's to indicate problems and missing attributes.

; ATTRIBUTOR: define void @wholly_unreachable()
; ATTRIBUTOR-NEXT: unreachable
define void @wholly_unreachable() {
  %a = load i32, i32* null
  ret void
}

; ATTRIBUTOR: define void @single_bb_unreachable(i1 %cond)
; ATTRIBUTOR-NEXT: br i1 %cond, label %t, label %e
; ATTRIBUTOR-EMPTY: 
; ATTRIBUTOR-NEXT: t:
; ATTRIBUTOR-NEXT: unreachable
; ATTRIBUTOR-EMPTY:
; ATTRIBUTOR-NEXT: e:
; ATTRIBUTOR-NEXT: ret void
define void @single_bb_unreachable(i1 %cond) {
  br i1 %cond, label %t, label %e
t:
  %b = load i32, i32* null
  br label %e
e:
  ret void
}

; ATTRIBUTOR: define void @null_pointer_is_defined()
; ATTRIBUTOR-NEXT: %a = load i32, i32* null
define void @null_pointer_is_defined() "null-pointer-is-valid"="true" {
  %a = load i32, i32* null
  ret void
}
