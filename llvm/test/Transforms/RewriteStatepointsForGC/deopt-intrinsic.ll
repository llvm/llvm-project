; RUN: opt -rewrite-statepoints-for-gc -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare i32 @llvm.experimental.deoptimize.i32(...)

define i32 @caller_0(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @caller_0(
; CHECK: @llvm.experimental.gc.statepoint.p0f_i32f(i64 2882400000, i32 0, i32 ()* @__llvm_deoptimize, i32 0
entry:
  %v = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 0, i32 addrspace(1)* %ptr) ]
  ret i32 %v
}


define i32 @caller_1(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @caller_1
; CHECK: @llvm.experimental.gc.statepoint.p0f_i32i32p1i32f(i64 2882400000, i32 0, i32 (i32, i32 addrspace(1)*)* bitcast (i32 ()* @__llvm_deoptimize to i32 (i32, i32 addrspace(1)*)*), i32 2, i32 0, i32 50, i32 addrspace(1)* %ptr
entry:
  %v = call i32(...) @llvm.experimental.deoptimize.i32(i32 50, i32 addrspace(1)* %ptr) [ "deopt"(i32 0) ]
  ret i32 %v
}
