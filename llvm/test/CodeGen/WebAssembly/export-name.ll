; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @test() #0 {
  ret void
}

declare void @test2() #1


attributes #0 = { "wasm-export-name"="foo" }
attributes #1 = { "wasm-export-name"="bar" }

; CHECK: .export_name test, foo
; CHECK: .export_name test2, bar
