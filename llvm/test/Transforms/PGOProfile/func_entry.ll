; RUN: llvm-profdata merge %S/Inputs/func_entry.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@s = common dso_local local_unnamed_addr global i32 0, align 4

define void @cold() {
; CHECK-LABEL: @cold()
; CHECK-SAME: #[[COLD_ATTR:[0-1]+]]
; CHECK-SAME: !prof ![[FUNC_ENTRY_COUNT_ZERO:[0-9]+]]

entry:
  store i32 1, i32* @s, align 4
  ret void
}

define void @hot() {
; CHECK-LABEL: @hot()
; CHECK-SAME: #[[HOT_ATTR:[0-1]+]]
; CHECK-SAME: !prof ![[FUNC_ENTRY_COUNT_NON_ZERO:[0-9]+]]
entry:
  %0 = load i32, i32* @s, align 4
  %add = add nsw i32 %0, 4
  store i32 %add, i32* @s, align 4
  ret void
}

define void @med() {
; CHECK-LABEL: @med
; CHECK-NOT: #
; CHECK-SAME: !prof ![[FUNC_ENTRY_COUNT_MED:[0-9]+]]

entry:
  store i32 1, i32* @s, align 4
  ret void
}

; CHECK-DAG: attributes #[[COLD_ATTR]] = { cold }
; CHECK-DAG: attributes #[[HOT_ATTR]] = { inlinehint }
; CHECK-DAG: ![[FUNC_ENTRY_COUNT_ZERO]] = !{!"function_entry_count", i64 10}
; CHECK-DAG: ![[FUNC_ENTRY_COUNT_NON_ZERO]] = !{!"function_entry_count", i64 9000}
; CHECK-DAG: ![[FUNC_ENTRY_COUNT_MED]] = !{!"function_entry_count", i64 50}
