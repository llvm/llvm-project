; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; "data" constant
@0 = private constant <{ i8, i8 }> <{i8 15, i8 11}>, section ".text"

; function-typed alias
@ud2 = alias void (), bitcast (<{ i8, i8 }>* @0 to void ()*)

; Check that "ud2" is emitted as a function symbol.
; CHECK: .type{{.*}}ud2,@function
