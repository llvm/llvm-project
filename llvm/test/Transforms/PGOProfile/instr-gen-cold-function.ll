; RUN: opt < %s  --passes=pgo-instr-gen -instrument-cold-function -S  | FileCheck --check-prefixes=COLD %s
; RUN: opt < %s  --passes=pgo-instr-gen -instrument-cold-function -instrument-cold-function-max-entry-count=1 -S  | FileCheck --check-prefixes=ENTRY-COUNT %s

; COLD: call void @llvm.instrprof.increment(ptr @__profn_foo, i64  [[#]], i32 1, i32 0)
; COLD-NOT: __profn_main

; ENTRY-COUNT: call void @llvm.instrprof.increment(ptr @__profn_foo, i64  [[#]], i32 1, i32 0)
; ENTRY-COUNT: call void @llvm.instrprof.increment(ptr @__profn_main, i64 [[#]], i32 1, i32 0)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() !prof !0 {
entry:
  ret void
}

define i32 @main() !prof !1 {
entry:
  ret i32 0
}

!0 = !{!"function_entry_count", i64 0}
!1 = !{!"function_entry_count", i64 1}
