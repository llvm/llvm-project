; RUN: opt -passes=def-use-instrumentation -S %s | FileCheck %s

declare void @nobodyfunc ()

define i32 @foo(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = mul i32 %a, 2
  ret i32 %b
}

; CHECK-LABEL: define i32 @foo(i32 %x)
; CHECK: call void @__def_use_trace_inst(i64 0)
; CHECK-NEXT: %a = add i32 %x, 1
; CHECK-NEXT: call void @__def_use_trace_inst(i64 1)
; CHECK-NEXT: call void @__def_use_trace_ssa_use(i64 0)
; CHECK-NEXT: %b = mul i32 %a, 2
; CHECK-NEXT: call void @__def_use_trace_inst(i64 2)
; CHECK-NEXT: call void @__def_use_trace_ssa_use(i64 1)
; CHECK-NEXT: ret i32 %b

; CHECK: declare void @__def_use_trace_inst(i64)
; CHECK: declare void @__def_use_trace_ssa_use(i64)