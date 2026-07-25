; RUN: opt -passes=def-use-instrumentation -S %s | FileCheck %s

declare void @nobodyfunc ()

define i32 @main() {
entry:
  %x = alloca i32
  ret i32 0
}

define i32 @foo() {
entry:
  %x = alloca i32
  ret i32 0
}

define i32 @bar() {
entry:
  %x = alloca i32
  ret i32 0
}

; CHECK: declare void @nobodyfunc()
; CHECK-LABEL: define i32 @main()
; CHECK: %x = alloca i32
; CHECK-NEXT: call void @__def_use_trace_enter()
; CHECK-NEXT: ret i32 0
; CHECK-LABEL: define i32 @foo()
; CHECK: %x = alloca i32
; CHECK-NEXT: call void @__def_use_trace_enter()
; CHECK-NEXT: ret i32 0
; CHECK-LABEL: define i32 @bar()
; CHECK: %x = alloca i32
; CHECK-NEXT: call void @__def_use_trace_enter()
; CHECK-NEXT: ret i32 0
; CHECK: declare void @__def_use_trace_enter()
