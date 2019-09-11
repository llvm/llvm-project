; RUN: llvm-profdata merge %S/Inputs/diag_no_value_sites.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s

; CHECK: Inconsistent number of value sites for memory intrinsic functions size profiling

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i8* %dst, i8* %src, i64 %n) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %n, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
