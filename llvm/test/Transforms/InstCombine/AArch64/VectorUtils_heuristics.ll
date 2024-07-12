; RUN: opt -S -passes=instcombine < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; This test checks that instcombine does not crash while invoking
; maskIsAllOneOrUndef, maskIsAllZeroOrUndef, or possiblyDemandedEltsInMask.

; CHECK-LABEL: novel_algorithm
; CHECK: unreachable
define void @novel_algorithm() {
entry:
  %a = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr undef, i32 1, <vscale x 16 x i1> splat (i1 true), <vscale x 16 x i8> undef)
  %b = add <vscale x 16 x i8> undef, %a
  call void @llvm.masked.store.nxv16i8.p0(<vscale x 16 x i8> %b, ptr undef, i32 1, <vscale x 16 x i1> splat (i1 true))
  unreachable
}

declare <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr, i32 immarg, <vscale x 16 x i1>, <vscale x 16 x i8>)

declare void @llvm.masked.store.nxv16i8.p0(<vscale x 16 x i8>, ptr, i32 immarg, <vscale x 16 x i1>)
