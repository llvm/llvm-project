; RUN: opt < %s  --passes=pgo-instr-gen -pgo-instrument-cold-function-only -pgo-function-entry-coverage -S  | FileCheck --check-prefixes=COLD %s
; RUN: opt < %s  --passes=pgo-instr-gen -pgo-instrument-cold-function-only -pgo-function-entry-coverage -pgo-cold-instrument-entry-threshold=1 -S  | FileCheck --check-prefixes=ENTRY-COUNT %s
; RUN: opt < %s  --passes=pgo-instr-gen -pgo-instrument-cold-function-only -pgo-function-entry-coverage -pgo-treat-unknown-as-cold -S  | FileCheck --check-prefixes=UNKNOWN-FUNC %s

; COLD: call void @llvm.instrprof.cover(ptr @__profn_foo, i64  [[#]], i32 1, i32 0)
; COLD-NOT: __profn_main
; COLD-NOT: __profn_bar

; ENTRY-COUNT: call void @llvm.instrprof.cover(ptr @__profn_foo, i64  [[#]], i32 1, i32 0)
; ENTRY-COUNT: call void @llvm.instrprof.cover(ptr @__profn_main, i64 [[#]], i32 1, i32 0)

; UNKNOWN-FUNC: call void @llvm.instrprof.cover(ptr @__profn_bar, i64  [[#]], i32 1, i32 0)
; UNKNOWN-FUNC: call void @llvm.instrprof.cover(ptr @__profn_foo, i64  [[#]], i32 1, i32 0)


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @bar() {
entry:
  ret void
}

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
