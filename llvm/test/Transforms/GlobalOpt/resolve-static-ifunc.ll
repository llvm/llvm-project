; RUN: opt --passes=globalopt -o - -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

$callee_with_trivial_resolver.resolver = comdat any
@callee_with_trivial_resolver.ifunc = weak_odr dso_local ifunc void (), ptr @callee_with_trivial_resolver.resolver
define weak_odr ptr @callee_with_trivial_resolver.resolver() comdat {
  ret ptr @callee_with_trivial_resolver._Msimd
}
define void @callee_with_trivial_resolver._Msimd() {
  ret void
}
define void @callee_with_trivial_resolver.default() {
  ret void
}

@unknown_condition = external global i1
$callee_with_complex_static_resolver.resolver = comdat any
@callee_with_complex_static_resolver.ifunc = weak_odr dso_local ifunc void (), ptr @callee_with_complex_static_resolver.resolver
define weak_odr ptr @callee_with_complex_static_resolver.resolver() comdat {
entry:
  %v = load i1, ptr @unknown_condition
  br i1 %v, label %fast, label %slow
fast:
  ret ptr @callee_with_complex_static_resolver._Msimd
slow:
  ret ptr @callee_with_complex_static_resolver._Msimd
}
define void @callee_with_complex_static_resolver._Msimd() {
  ret void
}
define void @callee_with_complex_static_resolver.default() {
  ret void
}

$callee_with_complex_dynamic_resolver.resolver = comdat any
@callee_with_complex_dynamic_resolver.ifunc = weak_odr dso_local ifunc void (), ptr @callee_with_complex_dynamic_resolver.resolver
define weak_odr ptr @callee_with_complex_dynamic_resolver.resolver() comdat {
entry:
  %v = load i1, ptr @unknown_condition
  br i1 %v, label %fast, label %slow
fast:
  ret ptr @callee_with_complex_dynamic_resolver._Msimd
slow:
  ret ptr @callee_with_complex_dynamic_resolver.default
}
define void @callee_with_complex_dynamic_resolver._Msimd() {
  ret void
}
define void @callee_with_complex_dynamic_resolver.default() {
  ret void
}

$callee_with_sideeffects_resolver.resolver = comdat any
@callee_with_sideeffects_resolver.ifunc = weak_odr dso_local ifunc void (), ptr @callee_with_sideeffects_resolver.resolver
define weak_odr ptr @callee_with_sideeffects_resolver.resolver() comdat {
  store i1 0, ptr @unknown_condition
  ret ptr @callee_with_sideeffects_resolver.default
}
define void @callee_with_sideeffects_resolver._Msimd() {
  ret void
}
define void @callee_with_sideeffects_resolver.default() {
  ret void
}

define void @caller() {
  call void @callee_with_trivial_resolver.ifunc()
  call void @callee_with_complex_static_resolver.ifunc()
  call void @callee_with_complex_dynamic_resolver.ifunc()
  call void @callee_with_sideeffects_resolver.ifunc()
  ret void
}

; CHECK-LABEL: define void @caller()
; CHECK-NEXT:     call void @callee_with_trivial_resolver._Msimd()
; CHECK-NEXT:     call void @callee_with_complex_static_resolver._Msimd()
; CHECK-NEXT:     call void @callee_with_complex_dynamic_resolver.ifunc()
; CHECK-NEXT:     call void @callee_with_sideeffects_resolver.ifunc()
; CHECK-NEXT:     ret void
