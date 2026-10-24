; With the internal copy, IPSCCP sees every call of @scale and propagates the
; constant argument and the resulting return value into the callers.
; RUN: opt -passes='default<O2>' -S < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -passes='default<O2>' -enable-linkonce-odr-internalization=likely-module-local -S < %s | FileCheck %s --check-prefix=ON

; OFF-LABEL: define {{.*}}i32 @caller(
; OFF: call i32 @scale(i32 5)
; OFF: call i32 @scale(i32 5)

; ON-LABEL: define {{.*}}i32 @caller(
; ON-NOT: call
; ON: ret i32 20
define i32 @caller() {
  %a = call i32 @scale(i32 5)
  %b = call i32 @scale(i32 5)
  %r = add i32 %a, %b
  ret i32 %r
}

define linkonce_odr i32 @scale(i32 %x) noinline "frontend-hint-likely-module-local" {
  %r = mul i32 %x, 2
  ret i32 %r
}
