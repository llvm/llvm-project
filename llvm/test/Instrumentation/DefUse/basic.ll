; RUN: opt -passes=def-use-instrumentation -S %s | FileCheck %s

define i32 @main() {
entry:
  ret i32 0
}

; CHECK: declare void @__def_use_trace_main_enter()