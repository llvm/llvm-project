; RUN: opt -S -wasm-add-missing-prototypes -o %t.ll %s 2>&1 | FileCheck %s -check-prefix=WARNING
; RUN: cat %t.ll | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; WARNING: warning: prototype-less function used with conflicting signatures: foo

; CHECK-LABEL: @call_with_conflicting_prototypes
; CHECK: %call1 = call i64 bitcast (i64 (i32, i32)* @foo to i64 (i32)*)(i32 42)
; CHECK: %call2 = call i64 @foo(i32 42, i32 43)
define void @call_with_conflicting_prototypes() {
  %call1 = call i64 bitcast (i64 (...)* @foo to i64 (i32)*)(i32 42)
  %call2 = call i64 bitcast (i64 (...)* @foo to i64 (i32, i32)*)(i32 42, i32 43)
  ret void
}

; CHECK: declare extern_weak i64 @foo(i32, i32)
declare extern_weak i64 @foo(...) #1

; CHECK-NOT: attributes {{.*}} = { {{.*}}"no-prototype"{{.*}} }
attributes #1 = { "no-prototype" }
