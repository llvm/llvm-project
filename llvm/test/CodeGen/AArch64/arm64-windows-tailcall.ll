; FIXME: Add tests for global-isel/fast-isel.

; RUN: llc < %s -mtriple=arm64-windows | FileCheck %s

%class.C = type { [1 x i32] }

define dso_local void @"?bar"(%class.C* inreg noalias sret %agg.result) {
entry:
; CHECK-LABEL: bar
; CHECK: mov x19, x0
; CHECK: bl "?foo"
; CHECK: mov x0, x19

  tail call void @"?foo"(%class.C* dereferenceable(4) %agg.result)
  ret void
}

declare dso_local void @"?foo"(%class.C* dereferenceable(4))
