;; Check that asan_globals is not marked large without an explicit code model.
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=asan -S | FileCheck %s

; CHECK: @__asan_global_global =
; CHECK-NOT: code_model "large"

@global = global i32 0, align 4