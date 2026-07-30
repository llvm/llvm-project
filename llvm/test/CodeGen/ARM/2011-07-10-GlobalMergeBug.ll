; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; CHECK-NOT: MergedGlobals

@a = internal unnamed_addr global i1 false
@b = internal global [64 x i8] zeroinitializer, align 64
