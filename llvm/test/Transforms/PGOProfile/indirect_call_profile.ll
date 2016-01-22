; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@bar = external global void ()*, align 8
; GEN: @__profn_foo = private constant [3 x i8] c"foo"

define void @foo() {
entry:
; GEN: entry:
; GEN-NEXT: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12884901887, i32 1, i32 0)
  %tmp = load void ()*, void ()** @bar, align 8
; GEN: [[ICALL_TARGET:%[0-9]+]] = ptrtoint void ()* %tmp to i64
; GEN-NEXT: call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12884901887, i64 [[ICALL_TARGET]], i32 0, i32 0)
  call void %tmp()
  ret void
}
