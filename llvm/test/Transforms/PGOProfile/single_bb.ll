; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: @__profn_single_bb = private constant [9 x i8] c"single_bb"

define i32 @single_bb() {
entry:
; GEN: entry:
; GEN: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_single_bb, i32 0, i32 0), i64 12884901887, i32 1, i32 0)
  ret i32 0
}
