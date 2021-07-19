; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

target triple = "wasm32-unknown-unknown"

define void @test() {
  call void @foo()
  call void @plain()
  ret void
}

declare void @foo() #0
declare void @plain()

attributes #0 = { "wasm-import-module"="bar" "wasm-import-name"="qux" }

; CHECK-NOT: .import_module plain
;     CHECK: .import_module foo, bar
;     CHECK: .import_name foo, qux
; CHECK-NOT: .import_module plain
