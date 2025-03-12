; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 -disable-cgp -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 -disable-cgp -verify-machineinstrs | %ptxas-verify %}

; Disable CGP which also folds branches, so that only BranchFolding is under
; the spotlight.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @foo(i32 %x, ptr %output) {
; CHECK-LABEL: .visible .func foo(
; CHECK-NOT: bra.uni
; CHECK-NOT: LBB0_
  %1 = icmp eq i32 %x, 1
  br i1 %1, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  store float 2.0, ptr %output
  ret void
}

; PR24299. no crash
define ptx_kernel void @hoge() #0 {
; CHECK-LABEL: .visible .entry hoge(
bb:
  br i1 undef, label %bb1, label %bb4

bb1:                                              ; preds = %bb1, %bb
  %tmp = phi i64 [ %tmp2, %bb1 ], [ undef, %bb ]
  %tmp2 = add nsw i64 %tmp, 1
  %tmp3 = icmp sle i64 %tmp, 0
  br i1 %tmp3, label %bb1, label %bb4

bb4:                                              ; preds = %bb4, %bb1, %bb
  br label %bb4
}
