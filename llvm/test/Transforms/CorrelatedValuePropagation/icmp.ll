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

; Make sure binary operator transfer functions are run when RHS is non-constant
; CHECK-LABEL: @test3
define i1 @test3(i32 %x, i32 %y) #0 {
entry:
  %cmp1 = icmp ult i32 %x, 10
  br i1 %cmp1, label %cont1, label %out

cont1:
  %cmp2 = icmp ult i32 %y, 10
  br i1 %cmp2, label %cont2, label %out

cont2:
  %add = add i32 %x, %y
  br label %cont3

cont3:
  %cmp3 = icmp ult i32 %add, 25
  br label %out

out:
  %ret = phi i1 [ true, %entry], [ true, %cont1 ], [ %cmp3, %cont3 ]
; CHECK: ret i1 true
  ret i1 %ret
}

; Same as previous but make sure nobody gets over-zealous
; CHECK-LABEL: @test4
define i1 @test4(i32 %x, i32 %y) #0 {
entry:
  %cmp1 = icmp ult i32 %x, 10
  br i1 %cmp1, label %cont1, label %out

cont1:
  %cmp2 = icmp ult i32 %y, 10
  br i1 %cmp2, label %cont2, label %out

cont2:
  %add = add i32 %x, %y
  br label %cont3

cont3:
  %cmp3 = icmp ult i32 %add, 15
  br label %out

out:
  %ret = phi i1 [ true, %entry], [ true, %cont1 ], [ %cmp3, %cont3 ]
; CHECK-NOT: ret i1 true
  ret i1 %ret
}

; Make sure binary operator transfer functions are run when RHS is non-constant
; CHECK-LABEL: @test5
define i1 @test5(i32 %x, i32 %y) #0 {
entry:
  %cmp1 = icmp ult i32 %x, 5
  br i1 %cmp1, label %cont1, label %out

cont1:
  %cmp2 = icmp ult i32 %y, 5
  br i1 %cmp2, label %cont2, label %out

cont2:
  %shifted = shl i32 %x, %y
  br label %cont3

cont3:
  %cmp3 = icmp ult i32 %shifted, 65536
  br label %out

out:
  %ret = phi i1 [ true, %entry], [ true, %cont1 ], [ %cmp3, %cont3 ]
; CHECK: ret i1 true
  ret i1 %ret
}

; Same as previous but make sure nobody gets over-zealous
; CHECK-LABEL: @test6
define i1 @test6(i32 %x, i32 %y) #0 {
entry:
  %cmp1 = icmp ult i32 %x, 5
  br i1 %cmp1, label %cont1, label %out

cont1:
  %cmp2 = icmp ult i32 %y, 15
  br i1 %cmp2, label %cont2, label %out

cont2:
  %shifted = shl i32 %x, %y
  br label %cont3

cont3:
  %cmp3 = icmp ult i32 %shifted, 65536
  br label %out

out:
  %ret = phi i1 [ true, %entry], [ true, %cont1 ], [ %cmp3, %cont3 ]
; CHECK-NOT: ret i1 true
  ret i1 %ret
}

attributes #4 = { noreturn }
