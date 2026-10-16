; RUN: opt < %s -passes='module(function(instsimplify),cgscc(inline))' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 @callee() {
  ret i32 42
}

@vt = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @callee to i64), i64 ptrtoint (ptr @vt to i64)) to i32)]

declare ptr @llvm.load.relative.i32(ptr, i32)

; CHECK-LABEL: define i32 @caller
; CHECK-NOT: call
; CHECK: ret i32 42
define i32 @caller() {
  %fn = call ptr @llvm.load.relative.i32(ptr @vt, i32 0)
  %res = call i32 %fn()
  ret i32 %res
}
