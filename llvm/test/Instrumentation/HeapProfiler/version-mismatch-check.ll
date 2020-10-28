; Check that the MemProf module constructor guards against compiler/runtime version
; mismatch.

; RUN: opt < %s -memprof-module -S | FileCheck %s
; RUN: opt < %s -memprof-module -memprof-guard-against-version-mismatch=0 -S | FileCheck %s --check-prefix=NOGUARD

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define internal void @memprof.module_ctor()
; CHECK:         call void @__memprof_version_mismatch_check_v1
; NOGUARD-NOT:   call void @__memprof_version_mismatch_check_
