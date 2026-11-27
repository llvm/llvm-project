; Manually add instcombine to ensure the hot/cold transformation happens before
; the LTO pipeline. The default LTO pipeline includes MemProfRemoveInfo which
; strips the memprof attributes unless the summary index indicates support.
; RUN: opt < %s -passes='function(instcombine),thinlto<O2>' -optimize-hot-cold-new -S | FileCheck %s
; RUN: opt < %s -passes='function(instcombine),lto<O2>' -optimize-hot-cold-new -S | FileCheck %s
; RUN: opt < %s -passes='function(instcombine),alloc-token' -optimize-hot-cold-new -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare ptr @_Znwm(i64)

define ptr @new_hot() sanitize_alloc_token {
; CHECK-LABEL: @new_hot(
; CHECK: call {{.*}} @__alloc_token__Znwm12__hot_cold_t(i64 10, i8 -2, i64 2689373973731826898){{.*}} !alloc_token
  %ret = call ptr @_Znwm(i64 10) #0, !alloc_token !0
  ret ptr %ret
}

attributes #0 = { builtin allocsize(0) "memprof"="hot" }
!0 = !{!"int", i1 false}
