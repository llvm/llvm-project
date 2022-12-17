; RUN: opt < %s -passes=inline -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @callee() alwaysinline {
; CHECK-LABEL: define ptr @callee()
    %1 = call ptr @llvm.strip.invariant.group.p0(ptr null)
    ret ptr %1
}

define ptr @caller() {
; CHECK-LABEL: define ptr @caller()
; CHECK-NEXT: call ptr @llvm.strip.invariant.group.p0(ptr null)
    %1 = call ptr @callee()
    ret ptr %1
}

declare ptr @llvm.strip.invariant.group.p0(ptr)
