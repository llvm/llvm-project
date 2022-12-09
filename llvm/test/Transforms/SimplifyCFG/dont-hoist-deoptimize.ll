; RUN: opt -S --passes='simplifycfg<hoist-common-insts>' -simplifycfg-hoist-common-skip-limit=0 %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.experimental.deoptimize.isVoid(...) #0

; REQUIRES: asserts
; XFAIL: *

define void @widget(i1 %arg) {
bb:
  br i1 %arg, label %bb1, label %bb4

bb1:                                              ; preds = %bb
  %tmp = trunc i64 5 to i32
  %tmp2 = trunc i64 0 to i32
  %tmp3 = trunc i64 0 to i32
  call void (...) @llvm.experimental.deoptimize.isVoid(i32 13) #0 [ "deopt"() ]
  ret void

bb4:                                              ; preds = %bb
  %tmp5 = trunc i64 5 to i32
  %tmp6 = trunc i64 1 to i32
  %tmp7 = trunc i64 0 to i32
  call void (...) @llvm.experimental.deoptimize.isVoid(i32 13) #0 [ "deopt"() ]
  ret void
}

attributes #0 = { nounwind }
