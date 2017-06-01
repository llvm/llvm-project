; RUN: opt %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: @__profn_statics_counter_naming.ll_func = private constant [30 x i8] c"statics_counter_naming.ll:func"

define internal i32 @func() {
entry:
  ret i32 0
}
