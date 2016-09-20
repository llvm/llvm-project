; Test that the coverage guards have proper comdat
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard  -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
$Foo = comdat any
; Function Attrs: uwtable
define linkonce_odr void @Foo() comdat {
entry:
  ret void
}

; CHECK: @__sancov_guard.Foo = linkonce_odr global i64 0, section "__sancov_guards", comdat($Foo)

