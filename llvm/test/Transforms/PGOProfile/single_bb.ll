; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_single_bb = private constant [9 x i8] c"single_bb"

define i32 @single_bb() {
entry:
; GEN: entry:
; GEN: call void @llvm.instrprof.increment(ptr @__profn_single_bb, i64 {{[0-9]+}}, i32 1, i32 0)
  ret i32 0
}
