; RUN: opt -correlated-propagation -S %s | FileCheck %s
; RUN: opt -passes=correlated-propagation -S %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: noreturn
declare void @check1(i1) #1

; Function Attrs: noreturn
declare void @check2(i1) #1

; Make sure we propagate the value of %tmp35 to the true/false cases
; CHECK-LABEL: @test1
; CHECK: call void @check1(i1 false)
; CHECK: call void @check2(i1 true)
define void @test1(i64 %tmp35) {
bb:
  %tmp36 = icmp sgt i64 %tmp35, 0
  br i1 %tmp36, label %bb_true, label %bb_false

bb_true:
  %tmp47 = icmp slt i64 %tmp35, 0
  tail call void @check1(i1 %tmp47) #4
  unreachable

bb_false:
  %tmp48 = icmp sle i64 %tmp35, 0
  tail call void @check2(i1 %tmp48) #4
  unreachable
}

; Function Attrs: noreturn
; This is the same as test1 but with a diamond to ensure we
; get %tmp36 from both true and false BBs.
; CHECK-LABEL: @test2
; CHECK: call void @check1(i1 false)
; CHECK: call void @check2(i1 true)
define void @test2(i64 %tmp35, i1 %inner_cmp) {
bb:
  %tmp36 = icmp sgt i64 %tmp35, 0
  br i1 %tmp36, label %bb_true, label %bb_false

bb_true:
  br i1 %inner_cmp, label %inner_true, label %inner_false

inner_true:
  br label %merge

inner_false:
  br label %merge

merge:
  %tmp47 = icmp slt i64 %tmp35, 0
  tail call void @check1(i1 %tmp47) #0
  unreachable

bb_false:
  %tmp48 = icmp sle i64 %tmp35, 0
  tail call void @check2(i1 %tmp48) #4
  unreachable
}

attributes #4 = { noreturn }
