; RUN: opt %s -S -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Note: this test mimics how the naming could have been after parsing in IR in
; a LLVMContext where some types were already available; but before the
; remangleIntrinsicFunctions has happened:
; The @llvm.ssa.copy intrinsics will have to be remangled.
; In certain cases (as shown here) the remangling can result in a name clash.
; This is also related to the llvm/test/tools/llvm-linker/remangle.ll testcase that checks
; a similar situation in the bitcode reader.

%fum = type { %aab, i8, [7 x i8] }
%aab = type { %aba }
%aba = type { [8 x i8] }
%fum.1 = type { %abb, i8, [7 x i8] }
%abb = type { %abc }
%abc = type { [4 x i8] }

declare void @foo(%fum*)

; Will be remagled to @"llvm.ssa.copy.p0p0s_fum.1s"
declare %fum.1** @"llvm.ssa.copy.p0p0s_fums"(%fum.1**)

; Will be remagled to @"llvm.ssa.copy.p0p0s_fums"
declare %fum** @"llvm.ssa.copy.p0p0s_fum.1s"(%fum**)

define void @foo1(%fum** %a, %fum.1 ** %b) {
  %b.copy = call %fum.1** @"llvm.ssa.copy.p0p0s_fums"(%fum.1** %b)
  %a.copy = call %fum** @"llvm.ssa.copy.p0p0s_fum.1s"(%fum** %a)
  ret void
}

define void @foo2(%fum.1 ** %b, %fum** %a) {
  %a.copy = call %fum** @"llvm.ssa.copy.p0p0s_fum.1s"(%fum** %a)
  %b.copy = call %fum.1** @"llvm.ssa.copy.p0p0s_fums"(%fum.1** %b)
  ret void
}

; CHECK-DAG: %fum = type { %aab, i8, [7 x i8] }
; CHECK-DAG: %aab = type { %aba }
; CHECK-DAG: %aba = type { [8 x i8] }
; CHECK-DAG: %fum.1 = type { %abb, i8, [7 x i8] }
; CHECK-DAG: %abb = type { %abc }
; CHECK-DAG: %abc = type { [4 x i8] }

; CHECK-LABEL: define void @foo1(%fum** %a, %fum.1** %b) {
; CHECK-NEXT:   %b.copy = call %fum.1** @llvm.ssa.copy.p0p0s_fum.1s(%fum.1** %b)
; CHECK-NEXT:   %a.copy = call %fum** @llvm.ssa.copy.p0p0s_fums(%fum** %a)
; CHECK-NEXT:  ret void

; CHECK-LABEL: define void @foo2(%fum.1** %b, %fum** %a) {
; CHECK-NEXT:   %a.copy = call %fum** @llvm.ssa.copy.p0p0s_fums(%fum** %a)
; CHECK-NEXT:  %b.copy = call %fum.1** @llvm.ssa.copy.p0p0s_fum.1s(%fum.1** %b)
; CHECK-NEXT:  ret void

; CHECK: declare %fum.1** @llvm.ssa.copy.p0p0s_fum.1s(%fum.1** returned)

; CHECK: declare %fum** @llvm.ssa.copy.p0p0s_fums(%fum** returned)
