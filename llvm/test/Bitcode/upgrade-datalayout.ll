; Test to make sure datalayout is automatically upgraded.
;
; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

