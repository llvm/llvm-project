; RUN: opt -passes=inline -S < %s | FileCheck %s

; Ensure `noipa` does _not_ control inlining. The `noipa` attribute controls
; _other_ interprocedural optimisations by affecting the result of
; `isDefinitionExact` and `hasExactDefinition` checks, but inlining is
; controlled separately by the `noinline` attribute.

define internal i32 @noipa(i32 %x) noipa {
  ret i32 %x
}

define internal i32 @noipa_noinline(i32 %x) noipa noinline {
  ret i32 %x
}

define i32 @bob() {
  %a = call i32 @noipa(i32 1)
  %b = call i32 @noipa_noinline(i32 %a)
  ret i32 %b
}

; CHECK-NOT: call i32 @noipa(
; CHECK: call i32 @noipa_noinline(
