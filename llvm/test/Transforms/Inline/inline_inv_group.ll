; RUN: opt < %s -inline -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @callee() alwaysinline {
; CHECK-LABEL: define i8* @callee()
    %1 = call i8* @llvm.strip.invariant.group.p0i8(i8* null)
    ret i8* %1
}

define i8* @caller() {
; CHECK-LABEL: define i8* @caller()
; CHECK-NEXT: call i8* @llvm.strip.invariant.group.p0i8(i8* null)
    %1 = call i8* @callee()
    ret i8* %1
}

declare i8* @llvm.strip.invariant.group.p0i8(i8*)
