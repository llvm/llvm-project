; RUN: opt -module-summary %s -o %t.bc

; RUN: llvm-dis %t.bc -o - | FileCheck %s

; Tests that caller is not eligible to import and it doesn't have refs to ifunc 'callee'

; CHECK: gv: (name: "caller", summaries: (function: ({{.*}}, flags: ({{.*}}notEligibleToImport: 1
; CHECK-NOT: refs
; CHECK-SAME: guid = 16677772384402303968

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__cpu_model = external global { i32, i32, i32, [1 x i32] }

@callee = internal ifunc void(), ptr @callee.resolver

define void @dispatch(ptr %func) {
    tail call void %func()
    ret void
}


define void @caller() {
entry:
  tail call void @dispatch(ptr @callee)
  ret void
}

define internal ptr @callee.resolver() {
resolver_entry:
  tail call void @__cpu_indicator_init()
  %0 = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i64 0, i32 3, i64 0)
  %1 = and i32 %0, 1024
  %.not = icmp eq i32 %1, 0
  %func_sel = select i1 %.not, ptr @callee.default.1, ptr @callee.avx2.0
  ret ptr %func_sel
}

define internal void @callee.default.1(i32 %a) {
  ret void
}

define internal void @callee.avx2.0(i32 %a) {
  ret void
}

declare i32 @rand()
declare void @__cpu_indicator_init()
