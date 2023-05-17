; RUN: llc -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

define ptr @test_call_return_type(i64 %size) {
entry:
; CHECK: bl xmalloc
  %0 = call noalias ptr @xmalloc(i64 undef)
  ret ptr %0
}

declare noalias ptr @xmalloc(i64)
