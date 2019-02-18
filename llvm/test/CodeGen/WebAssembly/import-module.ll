; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @test() {
  call void @foo()
  call void @plain()
  ret void
}

declare void @foo() #0
declare void @plain()

attributes #0 = { "wasm-import-module"="bar" }

; CHECK-NOT: .import_module plain
;     CHECK: .import_module foo, bar
; CHECK-NOT: .import_module plain
