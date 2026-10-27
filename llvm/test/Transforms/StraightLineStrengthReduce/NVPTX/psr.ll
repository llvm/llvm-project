; RUN: opt < %s -passes=slsr -enable-partial-strength-reduce -psr-threshold=1  -S | FileCheck %s
; RUN: llc < %s -march=nvptx64 -enable-partial-strength-reduce -psr-threshold=1 -mcpu=sm_35 | FileCheck %s --check-prefix=PTX

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

define void @foo(ptr nocapture readonly %in, ptr nocapture writeonly %out, i32 %n) {
; CHECK-LABEL: @foo(
; CHECK-LABEL: entry:
; CHECK: [[BASE:%.*]] = addrspacecast ptr %in to ptr addrspace(1)
; CHECK: [[MUL:%.*]] = mul  i32 %n, 2
; CHECK: [[ADD0:%.*]] = add i32 [[MUL]], 4
; CHECK: [[ADD1:%.*]] = add i32 [[ADD0]], [[MUL]]
; CHECK: [[A:%.*]] = sext i32 [[ADD1]] to i64
; CHECK: [[ROOT:%.*]] = getelementptr i32, ptr addrspace(1) [[BASE]], i64 [[A]]

; PTX-LABEL: .visible .func foo(
; PTX: ld.param.b64  [[IN:%.*]], [foo_param_0];
; PTX: cvta.to.global.u64  [[BASE:%.*]], [[IN]]
; PTX: add.s64  [[ROOT:%.*]], [[BASE]], [[_:%.*]]
entry:
  %0 = addrspacecast ptr %in to ptr addrspace(1)
  %cmp = icmp slt i32 %n, 0
  br i1 %cmp, label %if.then.us, label %for.inc16

; CHECK-LABEL: if.then.us:
; CHECK-NOT: sext i32 [[ADD1]]
; CHECK: load i32, ptr addrspace(1) [[ROOT]], align 4
; CHECK: [[GET1:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[ROOT]], i64 8
; CHECK: load i32, ptr addrspace(1) [[GET1]], align 4

; PTX: ld.global.b32 	[[_:%.*]], [[[ROOT]]+16];
; PTX: ld.global.b32 	[[_:%.*]], [[[ROOT]]+24];
if.then.us:                                       ; preds = %entry
  %mul1 = mul nsw i32 %n, 2
  %add0 = add nsw i32 %mul1, 4
  %add1 = add nsw i32 %add0, %mul1
  %1 = sext i32 %add1 to i64
  %getElem.us = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %1
  %tmp12.us = load i32, ptr addrspace(1) %getElem.us, align 4
  %2 = sext i32 %add1 to i64
  %3 = getelementptr i32, ptr addrspace(1) %0, i64 %2
  %getElem.us.111 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 8
  %tmp12.us.1 = load i32, ptr addrspace(1) %getElem.us.111, align 4
  %add14.us.1 = add nsw i32 %tmp12.us.1, %tmp12.us
  br label %for.inc16

for.inc16:                                        ; preds = %entry, %if.then.us
  %.us-phi = phi i32 [ %add14.us.1, %if.then.us ], [ 0, %entry ]
  %cmp7.1 = icmp slt i32 %n, 1
  br i1 %cmp7.1, label %if.then.us.18, label %for.inc16.1

; CHECK-LABEL: if.then.us.18:
; CHECK-NOT: sext i32 [[ADD1]]
; CHECK: load i32, ptr addrspace(1) [[ROOT]], align 4
; CHECK: [[GET2:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[ROOT]], i64 8
; CHECK: load i32, ptr addrspace(1) [[GET2]], align 4

; PTX: ld.global.b32 	[[_:%.*]], [[[ROOT]]+16];
; PTX: ld.global.b32 	[[_:%.*]], [[[ROOT]]+24];
if.then.us.18:                                    ; preds = %for.inc16
  %mul1.1 = mul nsw i32 %n, 2
  %add0.1 = add nsw i32 %mul1.1, 4
  %add1.1 = add nsw i32 %add0.1, %mul1.1
  %4 = sext i32 %add1.1 to i64
  %getElem.us.15 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %4
  %tmp12.us.16 = load i32, ptr addrspace(1) %getElem.us.15, align 4
  %add14.us.17 = add nsw i32 %tmp12.us.16, %.us-phi
  %5 = sext i32 %add1.1 to i64
  %6 = getelementptr i32, ptr addrspace(1) %0, i64 %5
  %getElem.us.1.113 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 8
  %tmp12.us.1.1 = load i32, ptr addrspace(1) %getElem.us.1.113, align 4
  %add14.us.1.1 = add nsw i32 %tmp12.us.1.1, %add14.us.17
  br label %for.inc16.1

for.inc16.1:                                      ; preds = %for.inc16, %if.then.us.18
  %.us-phi.1 = phi i32 [ %add14.us.1.1, %if.then.us.18 ], [ %.us-phi, %for.inc16 ]
  %7 = addrspacecast ptr %out to ptr addrspace(1)
  store i32 %.us-phi.1, ptr addrspace(1) %7, align 4
  ret void
}
