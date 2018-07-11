; RUN: opt -S -wasm-add-missing-prototypes %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@foo_addr = global i64 (i32)* bitcast (i64 (...)* @foo to i64 (i32)*), align 8

define void @bar(i32 %a) {
entry:
  %call = call i64 bitcast (i64 (...)* @foo to i64 (i32)*)(i32 42)
  ret void
}

declare i64 @foo(...) #1

attributes #1 = { "no-prototype" }

; CHECK: %call = call i64 @foo(i32 42)
; CHECK: declare i64 @foo(i32)
; CHECK-NOT: attributes {{.*}} = { {{.*}}"no-prototype"{{.*}} }
